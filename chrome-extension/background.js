/* background.js — service worker: toolbar toggle, right-click menu, and
   relay between content scripts and the offscreen model host. */

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

// Content scripts ask for a prediction; the offscreen host computes it.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.type !== "FD_PREDICT_REQUEST") return;
  (async () => {
    try {
      await ensureOffscreen();
      chrome.runtime.sendMessage(
        { target: "fd-offscreen", type: "FD_PREDICT", text: msg.text, head: msg.head },
        (resp) => {
          if (chrome.runtime.lastError) {
            sendResponse({ ok: false, error: chrome.runtime.lastError.message });
          } else {
            sendResponse(resp || { ok: false, error: "No response from model host" });
          }
        }
      );
    } catch (e) {
      sendResponse({ ok: false, error: (e && e.message) || String(e) });
    }
  })();
  return true; // keep the channel open for the async response
});
