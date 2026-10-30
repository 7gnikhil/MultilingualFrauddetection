/* background.js — service worker: toolbar toggle, right-click menu,
   relay between content scripts / popup and the offscreen model host,
   and the corrections log for the feedback-learning feature. */

const CORRECTIONS_KEY = "fdCorrections";
const MAX_CORRECTIONS = 500;

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "fraud-check-selection",
    title: "Check with Fraud Detector",
    contexts: ["selection"]
  });
  ensureOffscreen();
});

chrome.runtime.onStartup.addListener(() => { ensureOffscreen(); });

// Right-click on selected text -> send it to the page's floating panel
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "fraud-check-selection" && tab && tab.id) {
    chrome.tabs.sendMessage(tab.id, {
      type: "FRAUD_CHECK_TEXT",
      text: info.selectionText || ""
    });
  }
});

// Clicking the toolbar icon toggles the on-screen panel on the active tab
chrome.action.onClicked.addListener((tab) => {
  if (tab && tab.id) {
    chrome.tabs.sendMessage(tab.id, { type: "FRAUD_TOGGLE_PANEL" });
  }
});

// --- Offscreen model host -------------------------------------------------
let offscreenCreating = null;

async function ensureOffscreen() {
  try {
    if (await chrome.offscreen.hasDocument()) return;
    if (!offscreenCreating) {
      offscreenCreating = chrome.offscreen.createDocument({
        url: "offscreen.html",
        reasons: ["WORKERS"],
        justification:
          "Hosts the on-device fraud-detection model (ONNX Runtime Web) " +
          "so the trained model loads once and serves all tabs."
      });
    }
    await offscreenCreating;
  } catch (e) {
    // Already-created races land here; a live document is what matters.
    try { if (await chrome.offscreen.hasDocument()) return; } catch (e2) {}
    throw e;
  } finally {
    offscreenCreating = null;
  }
}

function sendToOffscreen(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (resp) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else if (!resp) {
        reject(new Error("No response from model host"));
      } else if (!resp.ok) {
        reject(new Error(resp.error || "Model host error"));
      } else {
        resolve(resp);
      }
    });
  });
}

async function getCorrections() {
  const o = await chrome.storage.local.get(CORRECTIONS_KEY);
  return Array.isArray(o[CORRECTIONS_KEY]) ? o[CORRECTIONS_KEY] : [];
}

// Content scripts / popup talk to the model host through here.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg) return;

  if (msg.type === "FD_PREDICT_REQUEST") {
    (async () => {
      try {
        await ensureOffscreen();
        const resp = await sendToOffscreen(
          { target: "fd-offscreen", type: "FD_PREDICT", text: msg.text, head: msg.head });
        sendResponse(resp);
      } catch (e) {
        sendResponse({ ok: false, error: (e && e.message) || String(e) });
      }
    })();
    return true;
  }

  // Feedback learning: the user corrected a verdict. The offscreen host
  // updates the head delta in real time; we log the correction here so
  // it can be counted, exported as CSV, and folded into future training.
  if (msg.type === "FD_FEEDBACK_REQUEST") {
    (async () => {
      try {
        await ensureOffscreen();
        const resp = await sendToOffscreen(
          { target: "fd-offscreen", type: "FD_LEARN", text: msg.text, head: msg.head, label: msg.label });
        const corrections = await getCorrections();
        corrections.push({
          text: String(msg.text || "").slice(0, 500),
          head: msg.head,
          label: msg.label ? 1 : 0,
          probabilityBefore: resp.result.probabilityBefore,
          probabilityAfter: resp.result.probability,
          ts: Date.now()
        });
        while (corrections.length > MAX_CORRECTIONS) corrections.shift();
        await chrome.storage.local.set({ [CORRECTIONS_KEY]: corrections });
        sendResponse({
          ok: true,
          result: {
            probability: resp.result.probability,
            threshold: resp.result.threshold,
            corrections: corrections.length
          }
        });
      } catch (e) {
        sendResponse({ ok: false, error: (e && e.message) || String(e) });
      }
    })();
    return true;
  }

  if (msg.type === "FD_LEARNING_STATE_REQUEST") {
    (async () => {
      const corrections = await getCorrections();
      const perHead = {};
      for (const c of corrections) perHead[c.head] = (perHead[c.head] || 0) + 1;
      sendResponse({ ok: true, corrections: corrections.length, perHead });
    })();
    return true;
  }

  if (msg.type === "FD_EXPORT_REQUEST") {
    (async () => {
      const corrections = await getCorrections();
      sendResponse({ ok: true, corrections });
    })();
    return true;
  }

  if (msg.type === "FD_RESET_LEARNING_REQUEST") {
    (async () => {
      try {
        await chrome.storage.local.set({ [CORRECTIONS_KEY]: [] });
        await ensureOffscreen();
        await sendToOffscreen({ target: "fd-offscreen", type: "FD_RESET_LEARNING" });
        sendResponse({ ok: true });
      } catch (e) {
        sendResponse({ ok: false, error: (e && e.message) || String(e) });
      }
    })();
    return true;
  }
});
