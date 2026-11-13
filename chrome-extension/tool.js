/* tool.js — full-page checker: paste/type/drag-drop, real on-device
 * verdicts, and the same correction learning as the on-page panel. */
(function () {
  "use strict";

  var VERDICTS = {
    fraud: { icon: "🚨", text: "FRAUD" },
    uncertain: { icon: "❓", text: "UNCERTAIN — NEEDS REVIEW" },
    safe: { icon: "✅", text: "LOOKS SAFE" }
  };

  var textEl = document.getElementById("text");
  var checkBtn = document.getElementById("check");
  var resultEl = document.getElementById("result");
  var dropEl = document.getElementById("drop");
  var pageDrop = document.getElementById("page-drop");
  var lastCheck = null;

  document.getElementById("open-settings").addEventListener("click", function () {
    if (chrome.runtime.openOptionsPage) chrome.runtime.openOptionsPage();
    else chrome.tabs.create({ url: chrome.runtime.getURL("options.html") });
  });
  document.getElementById("open-welcome").addEventListener("click", function () {
    chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
  });

  var nn = window.FraudNN.attach(document.getElementById("nn"));

  function bindDrop(el, onText) {
    ["dragenter", "dragover"].forEach(function (ev) {
      el.addEventListener(ev, function (e) { e.preventDefault(); el.classList.add("over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      el.addEventListener(ev, function (e) { e.preventDefault(); el.classList.remove("over"); });
    });
    el.addEventListener("drop", function (e) {
      var t = (e.dataTransfer && e.dataTransfer.getData("text/plain")) || "";
      if (t.trim()) onText(t);
    });
  }
  function acceptText(t) { textEl.value = t.trim(); runCheck(t); }
  bindDrop(dropEl, acceptText);

  var dragDepth = 0;
  window.addEventListener("dragenter", function (e) {
    e.preventDefault();
    if (e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], "text/plain") >= 0) {
      dragDepth++; pageDrop.classList.add("show");
    }
  });
  window.addEventListener("dragover", function (e) { e.preventDefault(); });
  window.addEventListener("dragleave", function (e) {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) pageDrop.classList.remove("show");
  });
  window.addEventListener("drop", function (e) {
    e.preventDefault();
    dragDepth = 0; pageDrop.classList.remove("show");
    var t = (e.dataTransfer && e.dataTransfer.getData("text/plain")) || "";
    if (t.trim() && e.target !== dropEl && !dropEl.contains(e.target)) acceptText(t);
  });

  function renderResult(res, lang) {
    var v = VERDICTS[res.label] || VERDICTS.uncertain;
    var pct = Math.round(res.confidence * 100);
    resultEl.className = "result show " + res.label;
    resultEl.innerHTML =
      "<div class='verdict'>" + v.icon + " " + v.text + "</div>" +
      "<div class='bar'><div style='width:" + pct + "%'></div></div>" +
      "<div class='meta'>Confidence: <b>" + pct + "%</b> · Language: <b>" + lang.label +
      "</b></div>" +
      "<div class='meta'>Model fraud probability: <b>" + Math.round(res.probability * 100) +
      "%</b> · Decision threshold: <b>" + Math.round(res.threshold * 100) + "%</b></div>" +
      (res.label === "uncertain"
        ? "<div class='reasons'>This score sits inside the ±" +
          Math.round((res.uncertaintyMargin || 0.08) * 100) +
          "-point uncertainty band around the threshold, so no confident verdict is shown.</div>" : "") +
      (res.reasons && res.reasons.length
        ? "<div class='reasons'>Signals: " + res.reasons.map(function (r) {
            return "“" + r + "”";
          }).join(", ") + "</div>" : "") +
      "<div class='feedback'><span>Wrong verdict? Teach it:</span>" +
      "<button class='fb-btn' data-label='1'>🚨 It's fraud</button>" +
      "<button class='fb-btn' data-label='0'>✅ It's safe</button></div>" +
      "<div class='learn-note' id='learn-note'></div>";
    resultEl.querySelectorAll(".fb-btn").forEach(function (b) {
      b.addEventListener("click", function () { onFeedback(parseInt(b.dataset.label, 10)); });
    });
    if (res.corrections) {
      var note = resultEl.querySelector("#learn-note");
      if (note) note.textContent = "🧠 Personal corrections active: " + res.corrections;
    }
  }

  async function onFeedback(label) {
    if (!lastCheck) return;
    var note = resultEl.querySelector("#learn-note");
    resultEl.querySelectorAll(".fb-btn").forEach(function (b) { b.disabled = true; });
    if (note) note.textContent = "🧠 Learning…";
    try {
      var out = await window.FraudModel.learn(
        lastCheck.text, lastCheck.lang, label, lastCheck.result);
      var lbl = window.FraudModel.verdictFrom(
        out.probability, out.threshold, out.uncertaintyMargin);
      var updated = {
        label: lbl,
        probability: out.probability,
        confidence: lbl === "safe" ? 1 - out.probability : out.probability,
        threshold: out.threshold,
        uncertaintyMargin: out.uncertaintyMargin || 0.08,
        modelVersion: out.modelVersion || (lastCheck.result && lastCheck.result.modelVersion),
        demo: false, reasons: [], corrections: out.corrections
      };
      lastCheck.result = updated;
      renderResult(updated, lastCheck.lang);
      var n2 = resultEl.querySelector("#learn-note");
      if (n2) n2.textContent = "🧠 Learned — the local model updated instantly. " +
        "Personal corrections so far: " + out.corrections +
        (out.sharedQueued ? " · An anonymized copy was queued for model improvement because sharing is ON." : "");
    } catch (err) {
      if (note) note.textContent = "Could not learn: " + (err && err.message ? err.message : "error");
      resultEl.querySelectorAll(".fb-btn").forEach(function (b) { b.disabled = false; });
    }
  }

  async function runCheck(raw) {
    var text = (raw || textEl.value || "").trim();
    if (!text) { textEl.focus(); return; }
    checkBtn.disabled = true; checkBtn.textContent = "Checking…";
    nn.scan();
    try {
      var lang = window.FraudDetectLanguage(text);
      var res = await window.FraudModel.predict(text, lang);
      lastCheck = { text: text, lang: lang, result: res };
      try {
        var st = await window.FraudModel.learningState();
        if (st && st.corrections) res.corrections = st.corrections;
      } catch (e) { /* cosmetic */ }
      nn.setVerdict(res.label, res.probability);
      renderResult(res, lang);
    } catch (err) {
      lastCheck = null;
      var detail = (err && err.message ? err.message : "Unknown error")
        .replace(/^MODEL UNAVAILABLE:\s*/i, "");
      resultEl.className = "result show unavailable";
      resultEl.innerHTML = "<div class='verdict'>⛔ MODEL UNAVAILABLE</div>" +
        "<div class='meta'>The trained model could not produce a verdict, so no " +
        "SAFE/FRAUD result is shown. " + detail + "</div>";
    } finally {
      checkBtn.disabled = false; checkBtn.textContent = "Check this message";
    }
  }
  checkBtn.addEventListener("click", function () { runCheck(); });
})();
