/* options.js — Settings page logic. Uses the exact same background
 * message API as the popup (FD_SETTINGS_*, FD_LEARNING_STATE_*,
 * FD_EXPORT_*, FD_RESET_*, FD_SYNC_NOW_*, FD_MODEL_*). */
(function () {
  "use strict";

  var countEl = document.getElementById("learn-count");
  var exportBtn = document.getElementById("export");
  var resetBtn = document.getElementById("reset");
  var shareEl = document.getElementById("share-corrections");
  var apiBaseEl = document.getElementById("api-base");
  var saveApiBtn = document.getElementById("save-api");
  var syncBtn = document.getElementById("sync-now");
  var queueEl = document.getElementById("queue-count");
  var settingsStatusEl = document.getElementById("settings-status");
  var modelVersionEl = document.getElementById("model-version");
  var modelStatusEl = document.getElementById("model-status");
  var checkModelBtn = document.getElementById("check-model");
  var applyModelBtn = document.getElementById("apply-model");
  var rollbackModelBtn = document.getElementById("rollback-model");

  document.getElementById("launch").addEventListener("click", function () {
    chrome.tabs.create({ url: chrome.runtime.getURL("tool.html") });
  });
  document.getElementById("open-welcome").addEventListener("click", function () {
    chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
  });

  function send(type, extra) {
    return new Promise(function (resolve) {
      chrome.runtime.sendMessage(Object.assign({ type: type }, extra || {}), function (resp) {
        resolve(chrome.runtime.lastError ? null : resp);
      });
    });
  }

  function setSettingsStatus(text) { settingsStatusEl.textContent = text || ""; }

  function renderModelState(state) {
    if (!state) return;
    modelVersionEl.textContent = state.activeVersion || "unknown";
    var parts = [];
    if (state.lastMessage) parts.push(state.lastMessage);
    else if (state.lastStatus) parts.push(state.lastStatus.replace(/_/g, " "));
    if (state.stagedVersion) parts.push("Staged: " + state.stagedVersion);
    if (state.hasPrevious) parts.push("Rollback available: " + (state.previousVersion || "previous"));
    if (state.lastCheckedAt) parts.push("Last checked: " + new Date(state.lastCheckedAt).toLocaleString());
    modelStatusEl.textContent = parts.join(" · ") || "Not checked yet.";
    applyModelBtn.disabled = !state.stagedVersion;
    rollbackModelBtn.disabled = !state.hasPrevious;
  }

  function renderSettings(resp) {
    if (!resp || !resp.ok) return;
    if (resp.settings) {
      shareEl.checked = resp.settings.shareCorrections === true;
      apiBaseEl.value = resp.settings.apiBaseUrl || "https://multilingual-frauddetection.vercel.app";
    }
    queueEl.textContent = typeof resp.queuedFeedback === "number" ? resp.queuedFeedback : 0;
    renderModelState(resp.modelState);
  }

  async function refreshSettings() { renderSettings(await send("FD_SETTINGS_GET_REQUEST")); }

  async function refreshCount() {
    var resp = await send("FD_LEARNING_STATE_REQUEST");
    var n = resp && resp.ok ? resp.corrections : 0;
    countEl.textContent = n;
    exportBtn.disabled = !n;
    resetBtn.disabled = !n && !(resp && resp.queuedFeedback);
    renderSettings(resp);
  }

  async function saveSettings(message) {
    saveApiBtn.disabled = true;
    shareEl.disabled = true;
    try {
      var resp = await send("FD_SETTINGS_SET_REQUEST", {
        settings: { shareCorrections: shareEl.checked, apiBaseUrl: apiBaseEl.value.trim() }
      });
      if (!resp || !resp.ok) {
        setSettingsStatus("Could not save: " + ((resp && resp.error) || "invalid API URL"));
        await refreshSettings();
        return;
      }
      renderSettings(resp);
      setSettingsStatus(message || (resp.settings.shareCorrections
        ? "Sharing is ON for future correction clicks."
        : "Sharing is OFF. Corrections stay on this device."));
    } finally {
      saveApiBtn.disabled = false;
      shareEl.disabled = false;
    }
  }

  shareEl.addEventListener("change", function () { saveSettings(); });
  saveApiBtn.addEventListener("click", function () { saveSettings("API settings saved."); });

  syncBtn.addEventListener("click", async function () {
    syncBtn.disabled = true;
    try {
      var resp = await send("FD_SYNC_NOW_REQUEST");
      if (!resp || !resp.ok) {
        setSettingsStatus("Sync failed. Queued corrections were kept.");
      } else if (resp.result && resp.result.reason === "sharing_off") {
        setSettingsStatus("Sharing is OFF, so nothing was uploaded.");
      } else if (resp.result && resp.result.reason === "backend_not_configured") {
        setSettingsStatus("Set a real backend URL first; queued corrections were kept.");
      } else {
        setSettingsStatus("Sync attempted. Remaining queued corrections: " + resp.queuedFeedback + ".");
      }
      await refreshSettings();
    } finally { syncBtn.disabled = false; }
  });

  checkModelBtn.addEventListener("click", async function () {
    checkModelBtn.disabled = true;
    try {
      var resp = await send("FD_MODEL_CHECK_REQUEST");
      if (!resp || !resp.ok) {
        modelStatusEl.textContent = "Update check failed: " + ((resp && resp.error) || "unknown error");
      } else {
        renderModelState(resp.modelState);
      }
    } finally { checkModelBtn.disabled = false; }
  });

  applyModelBtn.addEventListener("click", async function () {
    applyModelBtn.disabled = true;
    try {
      var resp = await send("FD_MODEL_APPLY_REQUEST");
      if (!resp || !resp.ok) {
        modelStatusEl.textContent = "Could not apply staged config: " + ((resp && resp.error) || "unknown error");
      } else {
        renderModelState(resp.modelState);
      }
    } finally { await refreshSettings(); }
  });

  rollbackModelBtn.addEventListener("click", async function () {
    if (!confirm("Roll back to the previous model config?")) return;
    rollbackModelBtn.disabled = true;
    try {
      var resp = await send("FD_MODEL_ROLLBACK_REQUEST");
      if (!resp || !resp.ok) {
        modelStatusEl.textContent = "Rollback failed: " + ((resp && resp.error) || "unknown error");
      } else {
        renderModelState(resp.modelState);
      }
    } finally { await refreshSettings(); }
  });

  function csvCell(v) { return '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"'; }

  exportBtn.addEventListener("click", async function () {
    var resp = await send("FD_EXPORT_REQUEST");
    if (!resp || !resp.ok || !resp.corrections.length) return;
    var lines = ["timestamp,language,corrected_label,model_probability_before,probability_after,message_anonymized"];
    resp.corrections.forEach(function (c) {
      lines.push([
        csvCell(new Date(c.ts).toISOString()),
        csvCell(c.head),
        csvCell(c.label ? "Yes" : "No"),
        csvCell(typeof c.probabilityBefore === "number" ? c.probabilityBefore.toFixed(4) : ""),
        csvCell(typeof c.probabilityAfter === "number" ? c.probabilityAfter.toFixed(4) : ""),
        csvCell(c.text)
      ].join(","));
    });
    var blob = new Blob([lines.join("\r\n")], { type: "text/csv" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "fraud-detector-corrections.csv";
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
  });

  resetBtn.addEventListener("click", async function () {
    if (!confirm("Reset personal learning? The model returns to the trained base model, the local corrections log is cleared, and any unsent queued feedback is discarded.")) return;
    await send("FD_RESET_LEARNING_REQUEST");
    await Promise.all([refreshCount(), refreshSettings()]);
  });

  Promise.all([refreshCount(), refreshSettings()]);
})();
