document.getElementById("toggle").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.id) {
    chrome.tabs.sendMessage(tab.id, { type: "FRAUD_TOGGLE_PANEL" });
    window.close();
  }
});

const countEl = document.getElementById("learn-count");
const exportBtn = document.getElementById("export");
const resetBtn = document.getElementById("reset");

function send(type) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type }, (resp) => {
      resolve(chrome.runtime.lastError ? null : resp);
    });
  });
}

async function refreshCount() {
  const resp = await send("FD_LEARNING_STATE_REQUEST");
  const n = resp && resp.ok ? resp.corrections : 0;
  countEl.textContent = n;
  exportBtn.disabled = !n;
  resetBtn.disabled = !n;
}

function csvCell(v) {
  return '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
}

exportBtn.addEventListener("click", async () => {
  const resp = await send("FD_EXPORT_REQUEST");
  if (!resp || !resp.ok || !resp.corrections.length) return;
  const lines = ["timestamp,language,corrected_label,model_probability_before,probability_after,message"];
  for (const c of resp.corrections) {
    lines.push([
      csvCell(new Date(c.ts).toISOString()),
      csvCell(c.head),
      csvCell(c.label ? "Yes" : "No"),
      csvCell(typeof c.probabilityBefore === "number" ? c.probabilityBefore.toFixed(4) : ""),
      csvCell(typeof c.probabilityAfter === "number" ? c.probabilityAfter.toFixed(4) : ""),
      csvCell(c.text)
    ].join(","));
  }
  const blob = new Blob([lines.join("\r\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "fraud-detector-corrections.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset personal learning? The model returns to exactly the trained base model and the corrections log is cleared.")) return;
  await send("FD_RESET_LEARNING_REQUEST");
  refreshCount();
});

refreshCount();
