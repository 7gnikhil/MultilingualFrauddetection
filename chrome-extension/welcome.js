/* welcome.js — launch page logic: Launch/Settings buttons, sample chips,
 * and a live scan that runs the REAL on-device model while the neural
 * network animation lights up stage by stage. */
(function () {
  "use strict";

  var SAMPLES = {
    fraud: "Congratulations! You have won a cash prize of Rs 50,000. Click the link and share your OTP to claim it now.",
    safe: "HDFC Bank: Rs. 2,500.00 debited from your A/c ending 4821 on 10-Nov-2026. Avl Bal: Rs. 18,240.55. Not you? Call 1800-266-4332."
  };
  var STAGES = [
    "Reading the message as <b>tokens</b>…",
    "<b>Anonymizing</b> — phone numbers, OTPs, links and accounts become placeholders…",
    "Detecting the <b>language</b> — Telugu, Hindi or English…",
    "Running the <b>MuRIL neural embedding</b>…",
    "Adding <b>lexical + structured</b> scam signals…",
    "Scoring with the <b>per-language classifier head</b>…",
    "Applying the decision <b>threshold</b>…"
  ];
  var VERDICTS = {
    fraud: { icon: "🚨", text: "FRAUD" },
    uncertain: { icon: "❓", text: "UNCERTAIN — NEEDS REVIEW" },
    safe: { icon: "✅", text: "LOOKS SAFE" }
  };

  document.getElementById("launch").addEventListener("click", function () {
    chrome.tabs.create({ url: chrome.runtime.getURL("tool.html") });
  });
  document.getElementById("open-settings").addEventListener("click", function () {
    if (chrome.runtime.openOptionsPage) chrome.runtime.openOptionsPage();
    else chrome.tabs.create({ url: chrome.runtime.getURL("options.html") });
  });

  var textEl = document.getElementById("demo-text");
  var resultEl = document.getElementById("demo-result");
  var stageEl = document.getElementById("stage-line");
  var runBtn = document.getElementById("run-scan");
  var defaultStage = stageEl.innerHTML;

  document.querySelectorAll(".chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
      textEl.value = SAMPLES[chip.dataset.sample] || "";
      resultEl.className = "demo-result";
      resultEl.innerHTML = "";
    });
  });

  var nn = window.FraudNN.attach(document.getElementById("nn"), {
    onStage: function (i) {
      if (STAGES[i]) stageEl.innerHTML = STAGES[i];
    }
  });

  runBtn.addEventListener("click", function () {
    var text = textEl.value.trim();
    if (!text) { textEl.focus(); return; }
    runBtn.disabled = true;
    runBtn.textContent = "Scanning…";
    resultEl.className = "demo-result";
    resultEl.innerHTML = "";

    var lang = window.FraudDetectLanguage(text);
    var prediction = window.FraudModel.predict(text, lang);
    prediction.catch(function () { /* handled after the animation */ });

    nn.scan(function () {
      prediction.then(function (res) {
        nn.setVerdict(res.label, res.probability);
        var v = VERDICTS[res.label] || VERDICTS.uncertain;
        resultEl.className = "demo-result show " + res.label;
        resultEl.innerHTML =
          "<div class='verdict'>" + v.icon + " " + v.text + "</div>" +
          "<div class='meta'>Fraud probability: <b>" + Math.round(res.probability * 100) +
          "%</b> · Decision threshold: <b>" + Math.round(res.threshold * 100) +
          "%</b> · Language: <b>" + lang.label + "</b> · Model v" +
          (res.modelVersion || "4.0.0") + " — computed on this device.</div>" +
          (res.reasons && res.reasons.length
            ? "<div class='meta'>Signals: " + res.reasons.map(function (r) {
                return "“" + r + "”";
              }).join(", ") + "</div>" : "");
        stageEl.innerHTML = defaultStage;
      }).catch(function (err) {
        resultEl.className = "demo-result show unavailable";
        resultEl.innerHTML =
          "<div class='verdict'>⛔ MODEL UNAVAILABLE</div>" +
          "<div class='meta'>The trained model could not produce a verdict, so no " +
          "SAFE/FRAUD result is shown. " +
          String((err && err.message) || err).replace(/^MODEL UNAVAILABLE:\s*/i, "") + "</div>";
        stageEl.innerHTML = defaultStage;
      }).finally(function () {
        runBtn.disabled = false;
        runBtn.textContent = "▶ Run live scan";
      });
    });
  });
})();
