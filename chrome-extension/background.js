/* background.js — service worker for Fraud Detector v1.3.0.
 *
 * Responsibilities:
 *  - toolbar/context-menu actions and offscreen model-host management;
 *  - the existing on-device feedback-learning relay;
 *  - a privacy-first, opt-in feedback sync queue; and
 *  - validation, staging, application, and rollback of downloaded model
 *    CONFIG files (the ONNX encoder is never replaced automatically).
 *
 * Privacy rule enforced here: raw correction text is used only in memory
 * to run the existing anonymizer. The persistent correction log and every
 * queued/uploaded feedback event contain anonymized text only.
 */

importScripts("anonymize.js");

const CORRECTIONS_KEY = "fdCorrections";
const SETTINGS_KEY = "fdFeedbackSettings";
const SYNC_QUEUE_KEY = "fdFeedbackSyncQueue";
const MODEL_STATE_KEY = "fdModelUpdateState";
const ACTIVE_CONFIG_KEY = "fdActiveModelConfig";
const STAGED_PACKAGE_KEY = "fdStagedModelPackage";
const PREVIOUS_PACKAGE_KEY = "fdPreviousModelPackage";

const MAX_CORRECTIONS = 500;
const MAX_QUEUE = 200;
const DEFAULT_API_BASE_URL = "https://multilingual-frauddetection.vercel.app";
const BUNDLED_MODEL_VERSION = "4.0.0";
const REQUIRED_LANGUAGES = ["English", "Hindi", "Telugu"];
const SYNC_ALARM = "fd-feedback-sync";

let syncInProgress = false;
let bundledConfigPromise = null;

function storageGet(keys) { return chrome.storage.local.get(keys); }
function storageSet(values) { return chrome.storage.local.set(values); }
function storageRemove(keys) { return chrome.storage.local.remove(keys); }

function extensionVersion() {
  try { return chrome.runtime.getManifest().version || "1.3.0"; }
  catch (e) { return "1.3.0"; }
}

function makeEventId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && crypto.getRandomValues) crypto.getRandomValues(bytes);
  else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function anonymizeForUpload(text) {
  if (typeof FraudAnonymize !== "function") {
    throw new Error("The anonymizer is unavailable; feedback was not queued or uploaded.");
  }
  return FraudAnonymize(String(text || ""));
}

function normalizeApiBaseUrl(value) {
  let raw = String(value || "").trim() || DEFAULT_API_BASE_URL;
  if (!/^https?:\/\//i.test(raw)) raw = "https://" + raw;
  const url = new URL(raw);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("API base URL must start with http:// or https://");
  }
  url.search = "";
  url.hash = "";
  url.pathname = url.pathname.replace(/\/+$/, "");
  return url.toString().replace(/\/$/, "");
}

function isPlaceholderApiBase(apiBaseUrl) {
  try { return new URL(apiBaseUrl).hostname.toLowerCase().startsWith("your-backend"); }
  catch (e) { return true; }
}

async function getSettings() {
  const stored = await storageGet(SETTINGS_KEY);
  const saved = stored[SETTINGS_KEY] || {};
  let apiBaseUrl = DEFAULT_API_BASE_URL;
  try { apiBaseUrl = normalizeApiBaseUrl(saved.apiBaseUrl || DEFAULT_API_BASE_URL); }
  catch (e) { /* fall back to the documented placeholder */ }
  return {
    // Deliberately defaults to OFF. Installing or updating must never
    // turn sharing on silently.
    shareCorrections: saved.shareCorrections === true,
    apiBaseUrl
  };
}

async function saveSettings(next) {
  const current = await getSettings();
  const settings = {
    shareCorrections: typeof next.shareCorrections === "boolean"
      ? next.shareCorrections : current.shareCorrections,
    apiBaseUrl: normalizeApiBaseUrl(
      typeof next.apiBaseUrl === "string" ? next.apiBaseUrl : current.apiBaseUrl)
  };
  await storageSet({ [SETTINGS_KEY]: settings });
  return settings;
}

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
  const o = await storageGet(CORRECTIONS_KEY);
  return Array.isArray(o[CORRECTIONS_KEY]) ? o[CORRECTIONS_KEY] : [];
}

// --- Opt-in feedback sync queue -------------------------------------------

async function getSyncQueue() {
  const o = await storageGet(SYNC_QUEUE_KEY);
  return Array.isArray(o[SYNC_QUEUE_KEY]) ? o[SYNC_QUEUE_KEY] : [];
}

async function saveSyncQueue(queue) {
  await storageSet({ [SYNC_QUEUE_KEY]: queue.slice(-MAX_QUEUE) });
}

function predictedLabelToBinary(predictedLabel, probability, threshold, probabilityBefore) {
  if (predictedLabel === 1 || predictedLabel === 0) return predictedLabel;
  if (typeof predictedLabel === "string") {
    const normalized = predictedLabel.toLowerCase();
    if (normalized === "fraud" || normalized === "1" || normalized === "yes") return 1;
    if (normalized === "safe" || normalized === "0" || normalized === "no") return 0;
  }
  const p = typeof probability === "number" ? probability : probabilityBefore;
  const t = typeof threshold === "number" ? threshold : 0.5;
  return typeof p === "number" && p >= t ? 1 : 0;
}

function buildFeedbackEvent(fields) {
  // This object is the complete upload contract. Do not add raw text,
  // page URLs, browser identifiers, or any other field to it.
  return {
    event_id: makeEventId(),
    text_anonymized: fields.anonymizedText,
    language: fields.head,
    head: fields.head,
    predicted_label: fields.predictedLabel,
    predicted_probability: fields.predictedProbability,
    corrected_label: fields.correctedLabel,
    reward: -1,
    model_version: fields.modelVersion || BUNDLED_MODEL_VERSION,
    extension_version: extensionVersion(),
    client_created_at: new Date().toISOString(),
    consent: true
  };
}

async function enqueueFeedbackEvent(event) {
  const queue = await getSyncQueue();
  if (queue.some((item) => item && item.event && item.event.event_id === event.event_id)) return queue;
  queue.push({ event, attempts: 0, queuedAt: Date.now(), nextAttemptAt: Date.now(), lastError: null });
  await saveSyncQueue(queue);
  processFeedbackQueue().catch(() => {});
  return queue;
}

function retryDelayMs(attempts) {
  const boundedAttempt = Math.min(Math.max(attempts || 0, 0), 8);
  return Math.min(15 * 60 * 1000, 5000 * Math.pow(2, boundedAttempt)) +
    Math.floor(Math.random() * 1000);
}

async function processFeedbackQueue() {
  if (syncInProgress) return { started: false, reason: "already_running" };
  const settings = await getSettings();
  if (!settings.shareCorrections) return { started: false, reason: "sharing_off" };
  if (isPlaceholderApiBase(settings.apiBaseUrl)) {
    return { started: false, reason: "backend_not_configured" };
  }

  syncInProgress = true;
  try {
    const queue = await getSyncQueue();
    const now = Date.now();
    const remaining = [];
    let changed = false;

    for (const item of queue) {
      if (!item || !item.event) { changed = true; continue; }
      if ((item.nextAttemptAt || 0) > now) { remaining.push(item); continue; }
      try {
        const response = await fetch(settings.apiBaseUrl + "/v1/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json", "Accept": "application/json" },
          body: JSON.stringify(item.event)
        });
        if (!response.ok) throw new Error("Feedback API returned HTTP " + response.status);
        changed = true; // Successful items are not copied to remaining.
      } catch (e) {
        item.attempts = (item.attempts || 0) + 1;
        item.nextAttemptAt = Date.now() + retryDelayMs(item.attempts);
        item.lastError = (e && e.message) || String(e);
        remaining.push(item);
        changed = true;
      }
    }
    if (changed) await saveSyncQueue(remaining);
    return { started: true, remaining: remaining.length };
  } finally {
    syncInProgress = false;
  }
}

function scheduleFeedbackSync() {
  if (chrome.alarms && chrome.alarms.create) {
    chrome.alarms.create(SYNC_ALARM, { periodInMinutes: 5 });
  }
  processFeedbackQueue().catch(() => {});
}

// --- Model config validation and updates -----------------------------------

function isFiniteNumberArray(value, expectedLength) {
  if (!Array.isArray(value)) return false;
  if (typeof expectedLength === "number" && value.length !== expectedLength) return false;
  return value.every((v) => typeof v === "number" && Number.isFinite(v));
}

function configVersion(config) {
  if (!config) return BUNDLED_MODEL_VERSION;
  const version = config.model_version || config.version || config.modelVersion;
  return version == null ? BUNDLED_MODEL_VERSION : String(version);
}

function validateModelConfig(config, expectedVersion) {
  const errors = [];
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    return { ok: false, errors: ["Config must be a JSON object."] };
  }
  const version = config.model_version || config.version || config.modelVersion;
  if (expectedVersion && version != null && String(version) !== String(expectedVersion)) {
    errors.push("Config version does not match the advertised model version.");
  }
  if (!config.languages || typeof config.languages !== "object") {
    return { ok: false, errors: errors.concat(["Config is missing its languages object."]) };
  }

  for (const language of REQUIRED_LANGUAGES) {
    const head = config.languages[language];
    if (!head || typeof head !== "object") { errors.push("Missing head for " + language + "."); continue; }
    if (typeof head.arm !== "string" || !head.arm) errors.push(language + " head is missing arm.");
    if (typeof head.threshold !== "number" || head.threshold < 0 || head.threshold > 1) {
      errors.push(language + " threshold must be between 0 and 1.");
    }
    if (typeof head.intercept !== "number" || !Number.isFinite(head.intercept)) {
      errors.push(language + " intercept must be a finite number.");
    }
    if (!isFiniteNumberArray(head.coef) || head.coef.length < 768) {
      errors.push(language + " coef must be a finite numeric array of at least 768 values.");
      continue;
    }
    if (!isFiniteNumberArray(head.scaler_e_mean, 768) ||
        !isFiniteNumberArray(head.scaler_e_scale, 768)) {
      errors.push(language + " embedding scaler arrays must each contain 768 finite numbers.");
    }
    const structLength = head.coef.length - 768;
    if (structLength > 0 &&
        (!isFiniteNumberArray(head.scaler_s_mean, structLength) ||
         !isFiniteNumberArray(head.scaler_s_scale, structLength))) {
      errors.push(language + " structured-feature scaler arrays do not match coef length.");
    }
    if (head.uncertainty_margin != null &&
        (typeof head.uncertainty_margin !== "number" ||
         head.uncertainty_margin < 0 || head.uncertainty_margin > 0.49)) {
      errors.push(language + " uncertainty_margin must be between 0 and 0.49.");
    }
    // v4 stack fields: hash and meta must arrive together or not at all.
    const hasHash = !!head.hash, hasMeta = !!head.meta;
    if (hasHash !== hasMeta) {
      errors.push(language + " stack must include both hash and meta (or neither).");
    } else if (hasHash) {
      const h = head.hash, m = head.meta;
      const validPairs = (pairs, n) => Array.isArray(pairs) && pairs.every((p) =>
        Array.isArray(p) && p.length === 2 && Number.isInteger(p[0]) &&
        p[0] >= 0 && p[0] < n && typeof p[1] === "number" && Number.isFinite(p[1]));
      if (!Number.isInteger(h.word_n_features) || h.word_n_features <= 0 ||
          !Number.isInteger(h.char_n_features) || h.char_n_features <= 0 ||
          typeof h.intercept !== "number" || !Number.isFinite(h.intercept) ||
          !validPairs(h.word_coef, h.word_n_features) ||
          !validPairs(h.char_coef, h.char_n_features)) {
        errors.push(language + " hash arm is malformed.");
      }
      if (typeof m.intercept !== "number" || !Number.isFinite(m.intercept) ||
          typeof m.weight_hash !== "number" || !Number.isFinite(m.weight_hash) ||
          typeof m.weight_int8 !== "number" || !Number.isFinite(m.weight_int8)) {
        errors.push(language + " meta stack must have finite intercept/weight_hash/weight_int8.");
      }
    }
  }
  if (config.uncertainty_margin != null &&
      (typeof config.uncertainty_margin !== "number" ||
       config.uncertainty_margin < 0 || config.uncertainty_margin > 0.49)) {
    errors.push("Global uncertainty_margin must be between 0 and 0.49.");
  }
  return { ok: errors.length === 0, errors };
}

function parseVersion(version) {
  const parts = String(version || "0").match(/\d+/g);
  return parts ? parts.map(Number) : [0];
}

function compareVersions(a, b) {
  const aa = parseVersion(a);
  const bb = parseVersion(b);
  const len = Math.max(aa.length, bb.length);
  for (let i = 0; i < len; i++) {
    const diff = (aa[i] || 0) - (bb[i] || 0);
    if (diff !== 0) return diff;
  }
  return 0;
}

async function getBundledConfig() {
  if (!bundledConfigPromise) {
    bundledConfigPromise = (async () => {
      const response = await fetch(chrome.runtime.getURL("model/config_v4.json"));
      if (!response.ok) throw new Error("Bundled model/config_v4.json could not be read.");
      const config = await response.json();
      const validation = validateModelConfig(config);
      if (!validation.ok) throw new Error("Bundled model config is invalid: " + validation.errors.join(" "));
      return config;
    })().catch((e) => { bundledConfigPromise = null; throw e; });
  }
  return bundledConfigPromise;
}

async function getActiveModelPackage() {
  const stored = await storageGet([ACTIVE_CONFIG_KEY, MODEL_STATE_KEY]);
  const activeConfig = stored[ACTIVE_CONFIG_KEY];
  if (activeConfig && validateModelConfig(activeConfig).ok) {
    const state = stored[MODEL_STATE_KEY] || {};
    return { version: state.activeVersion || configVersion(activeConfig), config: activeConfig, source: "downloaded" };
  }
  const bundled = await getBundledConfig();
  return { version: configVersion(bundled), config: bundled, source: "bundled" };
}

async function buildModelState() {
  const stored = await storageGet([MODEL_STATE_KEY, STAGED_PACKAGE_KEY, PREVIOUS_PACKAGE_KEY]);
  const saved = stored[MODEL_STATE_KEY] || {};
  let active = null;
  try { active = await getActiveModelPackage(); } catch (e) { /* checks surface this */ }
  const staged = stored[STAGED_PACKAGE_KEY];
  const previous = stored[PREVIOUS_PACKAGE_KEY];
  return {
    activeVersion: active ? active.version : (saved.activeVersion || BUNDLED_MODEL_VERSION),
    activeSource: active ? active.source : "unknown",
    stagedVersion: staged && staged.version ? String(staged.version) : null,
    hasPrevious: !!previous,
    previousVersion: previous && previous.version ? String(previous.version) : null,
    latestVersion: saved.latestVersion || null,
    lastCheckedAt: saved.lastCheckedAt || null,
    lastStatus: saved.lastStatus || "not_checked",
    lastMessage: saved.lastMessage || ""
  };
}

async function saveModelState(patch) {
  const stored = await storageGet(MODEL_STATE_KEY);
  const state = Object.assign({}, stored[MODEL_STATE_KEY] || {}, patch || {});
  await storageSet({ [MODEL_STATE_KEY]: state });
  return state;
}

async function checkForModelUpdate() {
  const settings = await getSettings();
  const checkedAt = new Date().toISOString();
  const active = await getActiveModelPackage();

  if (isPlaceholderApiBase(settings.apiBaseUrl)) {
    await saveModelState({
      activeVersion: active.version, lastCheckedAt: checkedAt,
      lastStatus: "backend_not_configured",
      lastMessage: "Set a real backend URL before checking for model updates."
    });
    return buildModelState();
  }

  const response = await fetch(settings.apiBaseUrl + "/v1/model/latest", {
    headers: { "Accept": "application/json" }
  });
  if (!response.ok) throw new Error("Model update API returned HTTP " + response.status);
  const latest = await response.json();
  const latestVersion = latest.model_version || latest.version || latest.modelVersion;
  if (!latestVersion) throw new Error("The latest-model response did not include a version.");

  if (compareVersions(latestVersion, active.version) <= 0) {
    await storageRemove(STAGED_PACKAGE_KEY);
    await saveModelState({
      activeVersion: active.version, latestVersion: String(latestVersion),
      lastCheckedAt: checkedAt, lastStatus: "up_to_date",
      lastMessage: "The active model config is current."
    });
    return buildModelState();
  }

  let config = latest.config || latest.model_config || null;
  let sourceUrl = latest.config_url || latest.configUrl || latest.download_url || latest.downloadUrl || null;
  if (!config && sourceUrl) {
    const configUrl = new URL(String(sourceUrl), settings.apiBaseUrl + "/");
    const configResponse = await fetch(configUrl.toString(), { headers: { "Accept": "application/json" } });
    if (!configResponse.ok) throw new Error("Model config download returned HTTP " + configResponse.status);
    config = await configResponse.json();
    sourceUrl = configUrl.toString();
  }

  if (!config) {
    await saveModelState({
      activeVersion: active.version, latestVersion: String(latestVersion),
      lastCheckedAt: checkedAt, lastStatus: "update_available_no_config",
      lastMessage: "A newer model was announced, but the response did not include a config file or config_url."
    });
    return buildModelState();
  }

  const validation = validateModelConfig(config, String(latestVersion));
  if (!validation.ok) {
    await saveModelState({
      activeVersion: active.version, latestVersion: String(latestVersion),
      lastCheckedAt: checkedAt, lastStatus: "invalid_config",
      lastMessage: "Downloaded config was rejected: " + validation.errors.join(" ")
    });
    return buildModelState();
  }

  await storageSet({
    [STAGED_PACKAGE_KEY]: {
      version: String(latestVersion), config, downloadedAt: checkedAt,
      sourceUrl: sourceUrl || (settings.apiBaseUrl + "/v1/model/latest")
    }
  });
  await saveModelState({
    activeVersion: active.version, latestVersion: String(latestVersion),
    lastCheckedAt: checkedAt, lastStatus: "staged",
    lastMessage: "Model config " + latestVersion + " downloaded and validated. Apply it from the popup when ready."
  });
  return buildModelState();
}

async function reloadOffscreenModel() {
  try {
    await ensureOffscreen();
    await sendToOffscreen({ target: "fd-offscreen", type: "FD_RELOAD_MODEL" });
  } catch (e) {
    // Storage is authoritative. If no live host can reload now, the active
    // config is read the next time the host is created.
  }
}

async function applyStagedModelUpdate() {
  const stored = await storageGet(STAGED_PACKAGE_KEY);
  const staged = stored[STAGED_PACKAGE_KEY];
  if (!staged || !staged.config) throw new Error("No validated staged model config is available.");
  const validation = validateModelConfig(staged.config, staged.version);
  if (!validation.ok) throw new Error("Staged config is invalid: " + validation.errors.join(" "));

  const active = await getActiveModelPackage();
  await storageSet({
    [PREVIOUS_PACKAGE_KEY]: {
      version: active.version,
      config: active.source === "bundled" ? null : active.config,
      source: active.source,
      savedAt: new Date().toISOString()
    },
    [ACTIVE_CONFIG_KEY]: staged.config
  });
  await storageRemove(STAGED_PACKAGE_KEY);
  await saveModelState({
    activeVersion: String(staged.version), latestVersion: String(staged.version),
    lastStatus: "applied",
    lastMessage: "Model config " + staged.version + " is active. The previous config is kept for rollback."
  });
  await reloadOffscreenModel();
  return buildModelState();
}

async function rollbackModelUpdate() {
  const stored = await storageGet(PREVIOUS_PACKAGE_KEY);
  const previous = stored[PREVIOUS_PACKAGE_KEY];
  if (!previous) throw new Error("No previous model config is available for rollback.");

  if (previous.config) {
    const validation = validateModelConfig(previous.config, previous.version);
    if (!validation.ok) throw new Error("Previous config is invalid: " + validation.errors.join(" "));
    await storageSet({ [ACTIVE_CONFIG_KEY]: previous.config });
  } else {
    await storageRemove(ACTIVE_CONFIG_KEY); // Null means the bundled config.
  }
  await storageRemove(PREVIOUS_PACKAGE_KEY);
  await saveModelState({
    activeVersion: String(previous.version || BUNDLED_MODEL_VERSION),
    lastStatus: "rolled_back",
    lastMessage: "Rolled back to model config " + (previous.version || BUNDLED_MODEL_VERSION) + "."
  });
  await reloadOffscreenModel();
  return buildModelState();
}

// --- Extension lifecycle ----------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "fraud-check-selection", title: "Check with Fraud Detector", contexts: ["selection"]
  });
  ensureOffscreen().catch(() => {});
  scheduleFeedbackSync();
});

chrome.runtime.onStartup.addListener(() => {
  ensureOffscreen().catch(() => {});
  scheduleFeedbackSync();
});

if (chrome.alarms && chrome.alarms.onAlarm) {
  chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm && alarm.name === SYNC_ALARM) processFeedbackQueue().catch(() => {});
  });
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "fraud-check-selection" && tab && tab.id) {
    chrome.tabs.sendMessage(tab.id, { type: "FRAUD_CHECK_TEXT", text: info.selectionText || "" });
  }
});

chrome.action.onClicked.addListener((tab) => {
  if (tab && tab.id) chrome.tabs.sendMessage(tab.id, { type: "FRAUD_TOGGLE_PANEL" });
});

// --- Message API --------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.target === "fd-offscreen") return;

  if (msg.type === "FD_PREDICT_REQUEST") {
    (async () => {
      try {
        await ensureOffscreen();
        sendResponse(await sendToOffscreen(
          { target: "fd-offscreen", type: "FD_PREDICT", text: msg.text, head: msg.head }));
      } catch (e) {
        sendResponse({ ok: false, error: (e && e.message) || String(e) });
      }
    })();
    return true;
  }

  if (msg.type === "FD_FEEDBACK_REQUEST") {
    (async () => {
      try {
        await ensureOffscreen();
        const resp = await sendToOffscreen({
          target: "fd-offscreen", type: "FD_LEARN",
          text: msg.text, head: msg.head, label: msg.label
        });

        const anonymizedText = anonymizeForUpload(msg.text);
        const correctedLabel = msg.label ? 1 : 0;
        const predictedProbability = typeof msg.predictedProbability === "number"
          ? msg.predictedProbability : resp.result.probabilityBefore;
        const threshold = typeof msg.threshold === "number" ? msg.threshold : resp.result.threshold;
        const predictedLabel = predictedLabelToBinary(
          msg.predictedLabel, predictedProbability, threshold, resp.result.probabilityBefore);
        const settings = await getSettings();

        let queuedEvent = null;
        // A correction click is the explicit feedback action. It is queued
        // only when the separate sharing switch is also ON. Same-label
        // confirmations are not corrections and are not uploaded.
        if (settings.shareCorrections && predictedLabel !== correctedLabel) {
          queuedEvent = buildFeedbackEvent({
            anonymizedText, head: msg.head, predictedLabel, predictedProbability,
            correctedLabel, modelVersion: msg.modelVersion || resp.result.modelVersion
          });
          await enqueueFeedbackEvent(queuedEvent);
        }

        const corrections = await getCorrections();
        corrections.push({
          text: anonymizedText, anonymized: true, head: msg.head, label: correctedLabel,
          predictedLabel, predictedProbability,
          probabilityBefore: resp.result.probabilityBefore,
          probabilityAfter: resp.result.probability,
          modelVersion: msg.modelVersion || resp.result.modelVersion || BUNDLED_MODEL_VERSION,
          sharedQueued: !!queuedEvent,
          eventId: queuedEvent ? queuedEvent.event_id : null,
          ts: Date.now()
        });
        while (corrections.length > MAX_CORRECTIONS) corrections.shift();
        await storageSet({ [CORRECTIONS_KEY]: corrections });
        const queue = await getSyncQueue();

        sendResponse({
          ok: true,
          result: {
            probability: resp.result.probability,
            threshold: resp.result.threshold,
            uncertaintyMargin: resp.result.uncertaintyMargin,
            modelVersion: resp.result.modelVersion,
            corrections: corrections.length,
            sharedQueued: !!queuedEvent,
            queuedFeedback: queue.length
          }
        });
      } catch (e) {
        sendResponse({ ok: false, error: (e && e.message) || String(e) });
      }
    })();
    return true;
  }

  if (msg.type === "FD_SETTINGS_GET_REQUEST") {
    (async () => {
      const [settings, queue, modelState] = await Promise.all([
        getSettings(), getSyncQueue(), buildModelState()
      ]);
      sendResponse({ ok: true, settings, queuedFeedback: queue.length, modelState });
    })();
    return true;
  }

  if (msg.type === "FD_SETTINGS_SET_REQUEST") {
    (async () => {
      try {
        const settings = await saveSettings(msg.settings || {});
        scheduleFeedbackSync();
        const [queue, modelState] = await Promise.all([getSyncQueue(), buildModelState()]);
        sendResponse({ ok: true, settings, queuedFeedback: queue.length, modelState });
      } catch (e) {
        sendResponse({ ok: false, error: (e && e.message) || String(e) });
      }
    })();
    return true;
  }

  if (msg.type === "FD_SYNC_NOW_REQUEST") {
    (async () => {
      const result = await processFeedbackQueue();
      const queue = await getSyncQueue();
      sendResponse({ ok: true, result, queuedFeedback: queue.length });
    })();
    return true;
  }

  if (msg.type === "FD_LEARNING_STATE_REQUEST") {
    (async () => {
      const [corrections, settings, queue, modelState] = await Promise.all([
        getCorrections(), getSettings(), getSyncQueue(), buildModelState()
      ]);
      const perHead = {};
      for (const c of corrections) perHead[c.head] = (perHead[c.head] || 0) + 1;
      sendResponse({ ok: true, corrections: corrections.length, perHead, settings, queuedFeedback: queue.length, modelState });
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
        // Reset also drops unsent queued events: users can withdraw local
        // feedback before it leaves the device.
        await storageSet({ [CORRECTIONS_KEY]: [], [SYNC_QUEUE_KEY]: [] });
        await ensureOffscreen();
        await sendToOffscreen({ target: "fd-offscreen", type: "FD_RESET_LEARNING" });
        sendResponse({ ok: true });
      } catch (e) {
        sendResponse({ ok: false, error: (e && e.message) || String(e) });
      }
    })();
    return true;
  }

  if (msg.type === "FD_MODEL_STATE_REQUEST") {
    (async () => {
      try { sendResponse({ ok: true, modelState: await buildModelState() }); }
      catch (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); }
    })();
    return true;
  }

  if (msg.type === "FD_MODEL_CHECK_REQUEST") {
    (async () => {
      try { sendResponse({ ok: true, modelState: await checkForModelUpdate() }); }
      catch (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); }
    })();
    return true;
  }

  if (msg.type === "FD_MODEL_APPLY_REQUEST") {
    (async () => {
      try { sendResponse({ ok: true, modelState: await applyStagedModelUpdate() }); }
      catch (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); }
    })();
    return true;
  }

  if (msg.type === "FD_MODEL_ROLLBACK_REQUEST") {
    (async () => {
      try { sendResponse({ ok: true, modelState: await rollbackModelUpdate() }); }
      catch (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); }
    })();
    return true;
  }
});
