/**
 * content.js — the on-screen experience Pathi asked for:
 *  1. A floating shield button that STAYS on screen (bottom-right by
 *     default), can be DRAGGED anywhere, and remembers its position.
 *  2. Clicking it opens the check panel.
 *  3. Into the panel you can DRAG selected text from the page, drop it,
 *     paste it, or type it — then Check.
 *  4. Selecting/highlighting text anywhere on the page pops a small
 *     "🛡️ Check" bubble next to the selection.
 *  5. Right-click → "Check with Fraud Detector" does the same.
 * Everything runs locally: anonymize -> detect language -> model -> verdict.
 */
(function () {
  "use strict";
  if (window.__fraudDetectorInjected) return;
  window.__fraudDetectorInjected = true;

  const POS_KEY = "fdFloatPos";

  // ---------- Floating button ----------
  const btn = document.createElement("button");
  btn.id = "fd-float-btn";
  btn.title = "Fraud Detector — click to open, drag to move";
  btn.innerHTML = "🛡️<span class='fd-dot'></span>";
  document.documentElement.appendChild(btn);

  try {
    const saved = JSON.parse(localStorage.getItem(POS_KEY) || "null");
    if (saved && typeof saved.x === "number") {
      btn.style.left = saved.x + "px"; btn.style.top = saved.y + "px";
      btn.style.right = "auto"; btn.style.bottom = "auto";
    }
  } catch (e) { /* first run */ }

  // Drag the button to move it; a click (no real drag) opens the panel.
  let drag = null, moved = false;
  btn.addEventListener("pointerdown", (e) => {
    const r = btn.getBoundingClientRect();
    drag = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    moved = false; btn.setPointerCapture(e.pointerId);
  });
  btn.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const x = Math.min(Math.max(e.clientX - drag.dx, 0), innerWidth - 56);
    const y = Math.min(Math.max(e.clientY - drag.dy, 0), innerHeight - 56);
    if (Math.abs(e.movementX) + Math.abs(e.movementY) > 0) moved = true;
    btn.style.left = x + "px"; btn.style.top = y + "px";
    btn.style.right = "auto"; btn.style.bottom = "auto";
    panel.style.left = Math.max(8, x - 284) + "px";
    panel.style.top = Math.max(8, y - panel.offsetHeight - 12) + "px";
    panel.style.right = "auto"; panel.style.bottom = "auto";
  });
  btn.addEventListener("pointerup", () => {
    if (drag && moved) {
      const r = btn.getBoundingClientRect();
      try { localStorage.setItem(POS_KEY, JSON.stringify({ x: r.left, y: r.top })); } catch (e) {}
    }
    drag = null;
    if (!moved) togglePanel();
  });

  // ---------- Panel ----------
  const panel = document.createElement("div");
  panel.id = "fd-panel";
  panel.style.display = "none";
  panel.innerHTML =
    "<div class='fd-head'><div><b>🛡️ Fraud Detector</b>" +
    "<small>Telugu · Hindi · English — runs on your device</small></div>" +
    "<button class='fd-close' title='Close'>✕</button></div>" +
    "<div class='fd-body'>" +
      "<div class='fd-drop' id='fd-drop'>Drag a message &amp; drop it here<br>" +
      "<strong>or select text on the page</strong></div>" +
      "<div class='fd-or'>— or paste / type —</div>" +
      "<textarea id='fd-text' placeholder='Paste the suspicious message here…'></textarea>" +
      "<button class='fd-check-btn' id='fd-check'>Check this message</button>" +
      "<div class='fd-result' id='fd-result'></div>" +
      "<div class='fd-privacy'>🔒 Checked locally in your browser. Nothing is sent anywhere.</div>" +
    "</div>";
  document.documentElement.appendChild(panel);

  const dropEl = panel.querySelector("#fd-drop");
  const textEl = panel.querySelector("#fd-text");
  const checkBtn = panel.querySelector("#fd-check");
  const resultEl = panel.querySelector("#fd-result");
  panel.querySelector(".fd-close").addEventListener("click", () => togglePanel(false));

  function togglePanel(force) {
    const show = typeof force === "boolean" ? force : panel.style.display === "none";
    panel.style.display = show ? "block" : "none";
    if (show) textEl.focus();
  }

  // ---------- Drag & drop into the panel ----------
  ["dragenter", "dragover"].forEach((ev) =>
    dropEl.addEventListener(ev, (e) => { e.preventDefault(); dropEl.classList.add("fd-over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropEl.addEventListener(ev, (e) => { e.preventDefault(); dropEl.classList.remove("fd-over"); }));
  dropEl.addEventListener("drop", (e) => {
    const dropped = (e.dataTransfer && e.dataTransfer.getData("text/plain")) || "";
    if (dropped.trim()) { textEl.value = dropped.trim(); runCheck(dropped); }
  });
  // Also accept a drag anywhere onto the floating button itself
  btn.addEventListener("dragover", (e) => e.preventDefault());
  btn.addEventListener("drop", (e) => {
    e.preventDefault();
    const dropped = (e.dataTransfer && e.dataTransfer.getData("text/plain")) || "";
    if (dropped.trim()) { togglePanel(true); textEl.value = dropped.trim(); runCheck(dropped); }
  });

  // ---------- Selection bubble ----------
  const bubble = document.createElement("button");
  bubble.id = "fd-bubble";
  bubble.textContent = "🛡️ Check if fraud";
  document.documentElement.appendChild(bubble);
  let selectedText = "";

  document.addEventListener("mouseup", (e) => {
    if (panel.contains(e.target) || e.target === btn || e.target === bubble) return;
    const sel = window.getSelection();
    const t = sel ? sel.toString().trim() : "";
    if (t.length >= 12) {
      selectedText = t;
      const r = sel.getRangeAt(0).getBoundingClientRect();
      bubble.style.left = Math.max(8, r.left) + "px";
      bubble.style.top = Math.max(8, r.top - 38) + "px";
      bubble.style.display = "block";
    } else {
      bubble.style.display = "none";
    }
  });
  document.addEventListener("mousedown", (e) => {
    if (e.target !== bubble) bubble.style.display = "none";
  });
  bubble.addEventListener("click", () => {
    bubble.style.display = "none";
    togglePanel(true);
    textEl.value = selectedText;
    runCheck(selectedText);
  });

  // ---------- Check flow: anonymize -> language -> model -> verdict ----------
  const VERDICTS = {
    fraud:  { icon: "🚨", text: "FRAUD" },
    likely: { icon: "⚠️", text: "LIKELY FRAUD — be careful" },
    safe:   { icon: "✅", text: "LOOKS SAFE" }
  };

  let lastCheck = null; // { text, lang } of the verdict on screen

  function renderResult(res, lang) {
    const v = VERDICTS[res.label] || VERDICTS.safe;
    const pct = Math.round(res.confidence * 100);
    resultEl.className = "fd-result show " + res.label;
    resultEl.innerHTML =
      "<div class='fd-verdict'>" + v.icon + " " + v.text +
      (res.demo ? "<span class='fd-demo-badge'>DEMO MODEL</span>" : "") + "</div>" +
      "<div class='fd-bar'><div style='width:" + pct + "%'></div></div>" +
      "<div class='fd-meta'>Confidence: <b>" + pct + "%</b> · Language: <b>" +
      lang.label + "</b></div>" +
      (res.reasons && res.reasons.length
        ? "<div class='fd-reasons'>Signals: " +
          res.reasons.map((r) => "“" + r + "”").join(", ") + "</div>" : "") +
      (res.demo
        ? "<div class='fd-reasons'>Keyword demo only — the trained on-device model could not be reached" +
          (res.fallbackError ? " (" + res.fallbackError + ")" : "") + ".</div>"
        : "<div class='fd-feedback'><span>Wrong verdict? Teach it:</span>" +
          "<button class='fd-fb-btn' data-label='1'>🚨 It's fraud</button>" +
          "<button class='fd-fb-btn' data-label='0'>✅ It's safe</button></div>" +
          "<div class='fd-learn-note' id='fd-learn-note'></div>");
    if (!res.demo) {
      resultEl.querySelectorAll(".fd-fb-btn").forEach((b) =>
        b.addEventListener("click", () => onFeedback(parseInt(b.dataset.label, 10))));
      if (res.corrections) {
        const note = resultEl.querySelector("#fd-learn-note");
        if (note) note.textContent = "🧠 Personal corrections active: " + res.corrections;
      }
    }
  }

  async function onFeedback(label) {
    if (!lastCheck) return;
    const note = resultEl.querySelector("#fd-learn-note");
    resultEl.querySelectorAll(".fd-fb-btn").forEach((b) => (b.disabled = true));
    if (note) note.textContent = "🧠 Learning…";
    try {
      const out = await window.FraudModel.learn(lastCheck.text, lastCheck.lang, label);
      const lbl = window.FraudModel.verdictFrom(out.probability, out.threshold);
      renderResult({
        label: lbl,
        probability: out.probability,
        confidence: lbl === "safe" ? 1 - out.probability : out.probability,
        threshold: out.threshold,
        demo: false,
        reasons: [],
        corrections: out.corrections
      }, lastCheck.lang);
      const n2 = resultEl.querySelector("#fd-learn-note");
      if (n2) n2.textContent = "🧠 Learned — the model updated instantly. " +
        "Personal corrections so far: " + out.corrections;
    } catch (err) {
      if (note) note.textContent = "Could not learn: " + (err && err.message ? err.message : "error");
      resultEl.querySelectorAll(".fd-fb-btn").forEach((b) => (b.disabled = false));
    }
  }

  async function runCheck(raw) {
    const text = (raw || textEl.value || "").trim();
    if (!text) { textEl.focus(); return; }
    checkBtn.disabled = true; checkBtn.textContent = "Checking…";
    try {
      const lang = window.FraudDetectLanguage(text);
      const res = await window.FraudModel.predict(text, lang);
      lastCheck = { text, lang };
      if (!res.demo) {
        try {
          const st = await window.FraudModel.learningState();
          if (st && st.corrections) res.corrections = st.corrections;
        } catch (e) { /* count is cosmetic */ }
      }
      renderResult(res, lang);
    } catch (err) {
      resultEl.className = "fd-result show likely";
      resultEl.innerHTML = "<div class='fd-verdict'>⚠️ Could not check</div>" +
        "<div class='fd-meta'>" + (err && err.message ? err.message : "Unknown error") + "</div>";
    } finally {
      checkBtn.disabled = false; checkBtn.textContent = "Check this message";
    }
  }
  checkBtn.addEventListener("click", () => runCheck());

  // ---------- Messages from background (toolbar / right-click) ----------
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "FRAUD_TOGGLE_PANEL") togglePanel();
    if (msg.type === "FRAUD_CHECK_TEXT" && msg.text) {
      togglePanel(true); textEl.value = msg.text; runCheck(msg.text);
    }
  });

  // ---------- First-run: ask to keep the button visible on screen ----------
  const ASK_KEY = "fdVisibleAsked";
  try {
    if (!localStorage.getItem(ASK_KEY)) {
      togglePanel(true);
      resultEl.className = "fd-result show likely";
      resultEl.innerHTML =
        "<div class='fd-verdict'>👋 Keep Fraud Detector on screen?</div>" +
        "<div class='fd-meta'>The 🛡️ button stays visible on every page so you can " +
        "click it, drag messages onto it, or select any text to check it. " +
        "You can drag the button anywhere. Nothing leaves your browser.</div>";
      localStorage.setItem(ASK_KEY, "1");
    }
  } catch (e) { /* storage unavailable on some pages */ }
})();
