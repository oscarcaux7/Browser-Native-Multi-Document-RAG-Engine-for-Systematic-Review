// UI controller. Heavy work lives in two workers:
//   rag-worker.js  PDF parsing, chunking, embeddings, hybrid search, IndexedDB
//   llm-worker.js  the chat model (WebLLM on WebGPU, or transformers.js on CPU)
import {
  GPU_MODELS, CPU_MODEL, EMBEDDING_MODELS, DEFAULT_SETTINGS, DEFAULT_GPU_MODEL, LIGHT_GPU_MODEL,
  findGpuModel, findEmbeddingModel,
} from "./config.js";
import { buildMessages, stripThinking, citedSources, isFollowUp } from "./prompt.js";
import { renderMarkdown, escapeHtml } from "./markdown.js";
import { openSource, closeViewer, step as viewerStep, isOpen as viewerOpen } from "./viewer.js";

const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = "synapse.settings";

// ------------------------------------------------------------------ settings
const settings = (() => {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
})();
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {}
}

const darkQuery = matchMedia("(prefers-color-scheme: dark)");
function applyTheme() {
  const root = document.documentElement;
  if (settings.theme === "system") delete root.dataset.theme;
  else root.dataset.theme = settings.theme;
}
const effectiveDark = () => (settings.theme === "system" ? darkQuery.matches : settings.theme === "dark");

// ---------------------------------------------------------------------- state
const caps = { webgpu: false, f16: false, adapter: "" };
const state = {
  docs: [],
  jobs: new Map(), // jobId -> {name, phase, done, total, error}
  embed: { status: "loading", text: "Starting…", device: null },
  llm: { status: "idle", model: null, progress: 0, text: "", cached: {} },
  history: [], // {question, answer}
  generating: null, // {requestId}
};
let readyWaiters = [];

// ------------------------------------------------------------------- helpers
function toast(text, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = text;
  $("toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 4000);
}
const fmtMB = (bytes) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`);
/** Turn low-level errors into something a user can act on. */
function friendlyError(msg) {
  if (/failed to fetch|network ?error|load failed|ERR_/i.test(msg))
    return "The model download was interrupted (network error). Check your connection and try again: files already downloaded stay cached.";
  if (/quota|storage/i.test(msg)) return "Not enough browser storage for this model. Free some space or pick a smaller model in Settings.";
  return msg;
}
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
let reqSeq = 0;
const nextId = () => `r${++reqSeq}`;

function setStatus(el, status, text) {
  el.classList.remove("state-loading", "state-ready", "state-error");
  if (status === "loading" || status === "ready" || status === "error") el.classList.add(`state-${status}`);
  el.querySelector(".txt, .model-chip-label").textContent = text;
}

// ------------------------------------------------------------ model choices
function autoModel() {
  const small = (navigator.deviceMemory && navigator.deviceMemory <= 4) || /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  return small ? LIGHT_GPU_MODEL : DEFAULT_GPU_MODEL;
}
/** The model id actually loaded: f16 build when the GPU supports shader-f16, else f32. */
function resolvedModelId() {
  if (!caps.webgpu) return CPU_MODEL.id;
  const spec = findGpuModel(settings.gpuModel || autoModel()) || findGpuModel(DEFAULT_GPU_MODEL);
  return caps.f16 ? spec.id : spec.f32;
}
function modelSpec(id = resolvedModelId()) {
  return caps.webgpu ? findGpuModel(id) : CPU_MODEL;
}
const modelLabel = (id) => modelSpec(id)?.label || id;

// --------------------------------------------------------------- RAG worker
const rag = new Worker(new URL("./rag-worker.js", import.meta.url), { type: "module" });
const ragPending = new Map();

function ragRequest(msg) {
  const requestId = nextId();
  return new Promise((resolve, reject) => {
    ragPending.set(requestId, { resolve, reject });
    rag.postMessage({ ...msg, requestId });
  });
}

rag.onmessage = ({ data: m }) => {
  switch (m.type) {
    case "booted":
      rag.postMessage({ type: "init", embeddingModel: settings.embeddingModel });
      break;
    case "model-progress": {
      const spec = findEmbeddingModel(settings.embeddingModel);
      const pct = m.total ? ` ${Math.round((m.loaded / m.total) * 100)}%` : "";
      state.embed = { status: "loading", text: `Loading ${spec.label}${pct}` };
      renderEmbedStatus();
      break;
    }
    case "ready":
      state.embed = { status: "ready", device: m.device, text: `${findEmbeddingModel(m.model).label} · ${m.device === "webgpu" ? "WebGPU" : "CPU"}` };
      state.docs = m.docs;
      renderEmbedStatus();
      renderLibrary();
      break;
    case "library":
      state.docs = m.docs;
      renderLibrary();
      break;
    case "ingest-progress":
      state.jobs.set(m.jobId, { name: m.name, phase: m.phase, done: m.done, total: m.total });
      scheduleLibraryRender();
      break;
    case "ingest-done":
      state.jobs.delete(m.jobId);
      if (!m.reindexed) toast(`Indexed ${m.name}: ${plural(m.pages, "page")}, ${plural(m.chunks, "passage")} in ${m.seconds.toFixed(1)} s`);
      renderLibrary();
      break;
    case "ingest-skip":
      state.jobs.delete(m.jobId);
      renderLibrary();
      toast(`${m.name} is already in your library`);
      break;
    case "ingest-error":
      state.jobs.set(m.jobId, { name: m.name, error: m.error });
      renderLibrary();
      setTimeout(() => {
        state.jobs.delete(m.jobId);
        renderLibrary();
      }, 8000);
      break;
    case "results":
      ragPending.get(m.requestId)?.resolve(m.sources);
      ragPending.delete(m.requestId);
      break;
    case "error":
      if (m.requestId && ragPending.has(m.requestId)) {
        ragPending.get(m.requestId).reject(new Error(m.error));
        ragPending.delete(m.requestId);
      } else if (m.during === "embedder" && m.model !== DEFAULT_SETTINGS.embeddingModel) {
        // An optional embedding model failed (unsupported device, network): go back to the default.
        toast(`${findEmbeddingModel(m.model).label} could not load here (${friendlyError(m.error)}). Switched back to the default embedding model.`, "error");
        settings.embeddingModel = DEFAULT_SETTINGS.embeddingModel;
        saveSettings();
        rag.postMessage({ type: "set-embedding", embeddingModel: settings.embeddingModel });
      } else {
        state.embed = { status: "error", text: `Embedding error: ${friendlyError(m.error)}` };
        renderEmbedStatus();
        toast(friendlyError(m.error), "error");
      }
      break;
  }
};
rag.onerror = (e) => {
  state.embed = { status: "error", text: "Could not start the indexing worker" };
  renderEmbedStatus();
  console.error(e);
};

// ---------------------------------------------------------------- LLM worker
let llm = null;
let llmBooted = null;
const llmListeners = new Map(); // requestId -> {onToken, resolve, reject}

function startLlmWorker() {
  const backend = caps.webgpu ? "webgpu" : "wasm";
  llm = new Worker(new URL(`./llm-worker.js?backend=${backend}`, import.meta.url), { type: "module" });
  llmBooted = new Promise((resolve) => {
    llm.onmessage = ({ data: m }) => {
      switch (m.type) {
        case "booted":
          resolve();
          break;
        case "cache-status":
          Object.assign(state.llm.cached, m.cached);
          renderModelList();
          break;
        case "load-progress":
          state.llm.progress = m.progress;
          state.llm.text = m.text;
          renderModelState();
          break;
        case "loaded":
          state.llm = { ...state.llm, status: "ready", model: m.model, progress: 1 };
          state.llm.cached[m.model] = true;
          renderModelState();
          renderModelList();
          readyWaiters.forEach((w) => w.resolve());
          readyWaiters = [];
          break;
        case "token":
          llmListeners.get(m.requestId)?.onToken(m.text);
          break;
        case "done":
          llmListeners.get(m.requestId)?.resolve(m.stats);
          llmListeners.delete(m.requestId);
          break;
        case "error":
          if (m.during === "load") {
            state.llm.status = "error";
            state.llm.text = m.error;
            renderModelState();
            readyWaiters.forEach((w) => w.reject(new Error(m.error)));
            readyWaiters = [];
            toast(`Model failed to load. ${friendlyError(m.error)}`, "error");
          } else if (m.fatal) {
            // GPU device lost: restart the worker; the model reloads from cache on the next question.
            llmListeners.get(m.requestId)?.reject(
              new Error("The GPU stopped responding (WebGPU device lost). The model will reload on your next question. If this repeats, pick a smaller model or fewer passages in Settings."),
            );
            llmListeners.delete(m.requestId);
            llm.terminate();
            state.llm = { ...state.llm, status: "idle", model: null, progress: 0 };
            startLlmWorker();
            renderModelState();
          } else if (m.requestId) {
            llmListeners.get(m.requestId)?.reject(new Error(m.error));
            llmListeners.delete(m.requestId);
          }
          break;
      }
    };
  });
}

async function loadModel() {
  const id = resolvedModelId();
  if (state.llm.status === "loading") return;
  if (state.llm.status === "ready" && state.llm.model === id) return;
  await llmBooted;
  state.llm = { ...state.llm, status: "loading", progress: 0, text: "Preparing…", model: id };
  renderModelState();
  llm.postMessage({ type: "load", model: id });
}

function modelReady() {
  if (state.llm.status === "ready" && state.llm.model === resolvedModelId()) return Promise.resolve();
  const p = new Promise((resolve, reject) => readyWaiters.push({ resolve, reject }));
  loadModel();
  return p;
}

/** Is the chosen model already in the browser cache? (no network) */
async function checkModelCache() {
  const id = resolvedModelId();
  if (caps.webgpu) {
    await llmBooted;
    llm.postMessage({ type: "cache-status", models: GPU_MODELS.flatMap((m) => [m.id, m.f32]) });
    return new Promise((resolve) => {
      const t = setInterval(() => {
        if (id in state.llm.cached) {
          clearInterval(t);
          resolve(state.llm.cached[id]);
        }
      }, 50);
      setTimeout(() => {
        clearInterval(t);
        resolve(false);
      }, 5000);
    });
  }
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    const hit = keys.some((r) => r.url.includes(CPU_MODEL.id) && r.url.includes(".onnx"));
    state.llm.cached[CPU_MODEL.id] = hit;
    return hit;
  } catch {
    return false;
  }
}

// ------------------------------------------------------------------ library
let libraryFrame = 0;
function scheduleLibraryRender() {
  if (!libraryFrame) libraryFrame = requestAnimationFrame(() => ((libraryFrame = 0), renderLibrary()));
}

function renderLibrary() {
  const ul = $("library");
  const items = [];
  for (const d of state.docs) {
    const job = state.jobs.get(d.id); // re-indexing after an embedding-model change
    const meta = job && !job.error ? `Re-indexing · ${job.phase} ${job.done}/${job.total}` : `${plural(d.pages, "page")} · ${plural(d.chunks, "passage")}`;
    items.push(`<li class="doc${job ? " busy" : ""}" data-id="${d.id}">
      <svg aria-hidden="true"><use href="#i-file"/></svg>
      <span class="doc-name" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>
      <span class="doc-meta">${meta}</span>
      <button class="icon-btn" data-remove="${d.id}" aria-label="Remove ${escapeHtml(d.name)}" title="Remove"><svg><use href="#i-trash"/></svg></button>
    </li>`);
  }
  for (const [jobId, j] of state.jobs) {
    if (state.docs.some((d) => d.id === jobId)) continue;
    if (j.error) {
      items.push(`<li class="doc failed"><svg aria-hidden="true"><use href="#i-file"/></svg>
        <span class="doc-name">${escapeHtml(j.name)}</span><span class="doc-meta">${escapeHtml(j.error)}</span></li>`);
      continue;
    }
    const phase = { queued: "Waiting", reading: "Reading", parsing: "Extracting text", embedding: "Embedding" }[j.phase] || "Working";
    const unit = j.phase === "parsing" ? "pages" : "passages";
    const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
    const weight = j.phase === "embedding" ? 0.25 + 0.75 * (pct / 100) : j.phase === "parsing" ? 0.25 * (pct / 100) : 0;
    const detail = j.total && j.phase !== "reading" ? ` · ${j.done}/${j.total} ${unit}` : "";
    items.push(`<li class="doc busy">
      <svg aria-hidden="true"><use href="#i-file"/></svg>
      <span class="doc-name">${escapeHtml(j.name)}</span>
      <span class="doc-meta">${phase}${detail}</span>
      <button class="icon-btn" data-cancel="${jobId}" aria-label="Cancel indexing" title="Cancel"><svg><use href="#i-x"/></svg></button>
      <div class="bar" role="progressbar" aria-valuenow="${Math.round(weight * 100)}" aria-valuemin="0" aria-valuemax="100"><i style="width:${(weight * 100).toFixed(1)}%"></i></div>
    </li>`);
  }
  ul.innerHTML = items.join("") || `<li class="empty-lib">No documents yet.</li>`;
  const pages = state.docs.reduce((s, d) => s + d.pages, 0);
  $("library-count").textContent = state.docs.length ? `${plural(state.docs.length, "doc")} · ${pages} p.` : "";
  renderSteps();
}

function renderEmbedStatus() {
  setStatus($("embed-status"), state.embed.status, state.embed.text);
}

function addFiles(fileList) {
  const files = [...fileList].filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
  const skipped = fileList.length - files.length;
  if (skipped) toast(`${plural(skipped, "file")} skipped: only PDFs are supported`);
  if (!files.length) return;
  const jobs = files.map((file) => ({ file, jobId: nextId() }));
  for (const { file, jobId } of jobs) state.jobs.set(jobId, { name: file.name, phase: "queued", done: 0, total: 0 });
  renderLibrary();
  rag.postMessage({ type: "ingest", files: jobs });
}

// --------------------------------------------------------------- model state
function renderModelState() {
  const chip = $("model-chip");
  const id = resolvedModelId();
  const l = state.llm;
  const backend = caps.webgpu ? "WebGPU" : "CPU";
  if (l.status === "loading") setStatus(chip, "loading", `${modelLabel(l.model)} · ${Math.round(l.progress * 100)}%`);
  else if (l.status === "ready") setStatus(chip, "ready", `${modelLabel(l.model)} · ${backend}`);
  else if (l.status === "error") setStatus(chip, "error", `${modelLabel(id)} failed to load`);
  else setStatus(chip, "", `${modelLabel(id)} · not loaded`);
  renderSteps();
  // live progress inside a pending answer
  document.querySelectorAll("[data-loading-model]").forEach((el) => {
    el.textContent = l.status === "loading" ? `Loading ${modelLabel(l.model)}… ${Math.round(l.progress * 100)}%` : el.textContent;
  });
}

function renderSteps() {
  const hasDocs = state.docs.length > 0;
  const ready = state.llm.status === "ready";
  $("step-docs").classList.toggle("done", hasDocs);
  $("load-samples").hidden = hasDocs || state.jobs.size > 0;
  $("step-model").classList.toggle("done", ready);
  const cta = $("load-model-cta");
  const spec = modelSpec();
  if (ready) {
    $("step-model-text").textContent = `${modelLabel(state.llm.model)} is ready on ${caps.webgpu ? "your GPU (WebGPU)" : "your CPU (WebAssembly)"}.`;
    cta.hidden = true;
  } else if (state.llm.status === "loading") {
    $("step-model-text").textContent = state.llm.text || "Loading…";
    cta.hidden = true;
  } else {
    const cached = state.llm.cached[resolvedModelId()];
    $("step-model-text").textContent = caps.webgpu
      ? "Downloaded once, then cached by your browser. Runs on your GPU via WebGPU."
      : "WebGPU is not available here, so a smaller model runs on the CPU. It works, but slowly.";
    cta.hidden = false;
    cta.textContent = cached ? `Load ${spec.label}` : `Download ${spec.label} (${spec.download})`;
  }
  $("suggestions").hidden = !hasDocs;
}

// ---------------------------------------------------------------------- chat
const msgSources = new WeakMap(); // message element -> sources
let activeCite = null;

function scrollToBottom(force = false) {
  const c = $("chat");
  const near = c.scrollHeight - c.scrollTop - c.clientHeight < 140;
  if (force || near) c.scrollTop = c.scrollHeight;
}

function sourceButton(s, n, cited) {
  const snippet = s.text.length > 220 ? `${s.text.slice(0, 220)}…` : s.text;
  return `<button type="button" class="source${cited ? "" : " uncited"}" data-source="${n}">
    <span class="cite n" aria-hidden="true">${n}</span>
    <span class="source-title">${escapeHtml(s.docName)} <span class="pg">· p. ${s.page}</span></span>
    <span class="source-snippet">${escapeHtml(snippet)}</span>
  </button>`;
}

function renderSources(el, sources, answer) {
  const cited = citedSources(answer, sources.length);
  const citedSet = new Set(cited);
  const rest = sources.map((_, i) => i + 1).filter((n) => !citedSet.has(n));
  let html = "";
  if (cited.length) {
    html += `<div class="sources-head">Sources</div>` + cited.map((n) => sourceButton(sources[n - 1], n, true)).join("");
  }
  if (rest.length) {
    const label = cited.length ? `${plural(rest.length, "other passage")} retrieved` : `${plural(rest.length, "passage")} retrieved (none cited)`;
    html += `<details class="more-sources"${cited.length ? "" : " open"}><summary>${label}</summary><div>${rest
      .map((n) => sourceButton(sources[n - 1], n, false))
      .join("")}</div></details>`;
  }
  el.innerHTML = html;
}

async function ask(question, { overview = false } = {}) {
  if (state.generating) return;
  if (!state.docs.length) {
    toast("Add at least one PDF first.");
    return;
  }
  if (state.embed.status !== "ready") {
    toast("The embedding model is still loading.");
    return;
  }
  $("empty").hidden = true;
  const shownQuestion = overview ? "Give me an overview of all documents." : question;
  const user = document.createElement("div");
  user.className = "msg-user";
  user.textContent = shownQuestion;
  const ai = document.createElement("div");
  ai.className = "msg-ai";
  ai.innerHTML = `<div class="answer"><span class="thinking">Searching your documents…</span></div><div class="sources"></div><div class="meta-line"></div>`;
  $("messages").append(user, ai);
  scrollToBottom(true);

  const answerEl = ai.querySelector(".answer");
  const requestId = nextId();
  state.generating = { requestId };
  setComposerBusy(true);
  const t0 = performance.now();

  try {
    const topK = caps.webgpu ? settings.topK : Math.min(settings.topK, 4);
    const sources = overview
      ? await ragRequest({ type: "search", mode: "overview", perDocChars: Math.floor(5200 / Math.max(state.docs.length, 1)) })
      : await ragRequest({
          type: "search",
          // follow-ups ("what about its power draw?") are searched together with the previous question
          query: isFollowUp(question, state.history) ? `${state.history.at(-1).question} ${question}` : question,
          topK,
        });
    msgSources.set(ai, sources);

    if (state.llm.status !== "ready" || state.llm.model !== resolvedModelId()) {
      answerEl.innerHTML = `<span class="thinking" data-loading-model>Loading ${escapeHtml(modelLabel(resolvedModelId()))}…</span>`;
      await modelReady();
    }
    if (!state.generating || state.generating.requestId !== requestId) throw new Error("stopped");
    answerEl.innerHTML = `<span class="thinking">Reading ${plural(sources.length, "passage")}…</span>`;

    const threshold = findEmbeddingModel(settings.embeddingModel).lowRelevance;
    const best = Math.max(...sources.map((s) => s.similarity ?? 1));
    const lowRelevance = !overview && threshold != null && sources.length > 0 && best < threshold;
    const messages = buildMessages({ question, sources, history: state.history, mode: overview ? "overview" : "qa", lowRelevance });
    let raw = "";
    let frame = 0;
    let finished = false;
    const paint = () => {
      frame = 0;
      if (finished) return;
      const visible = stripThinking(raw);
      answerEl.innerHTML = visible
        ? renderMarkdown(visible, sources.length) + `<span class="caret"></span>`
        : `<span class="thinking">Thinking…</span>`;
      scrollToBottom();
    };
    const stats = await new Promise((resolve, reject) => {
      llmListeners.set(requestId, {
        onToken: (t) => {
          raw += t;
          if (!frame) frame = requestAnimationFrame(paint);
        },
        resolve,
        reject,
      });
      llm.postMessage({
        type: "generate",
        requestId,
        messages,
        temperature: settings.temperature,
        maxTokens: settings.maxTokens,
        disableThinking: !!modelSpec()?.thinking,
      });
    });
    finished = true;
    cancelAnimationFrame(frame);
    const answer = stripThinking(raw).trim() || "_(no answer)_";
    answerEl.innerHTML = renderMarkdown(answer, sources.length);
    renderSources(ai.querySelector(".sources"), sources, answer);
    const meta = [modelLabel(state.llm.model), `${((performance.now() - t0) / 1000).toFixed(1)} s`];
    if (stats?.ttft) meta.push(`first token ${stats.ttft.toFixed(1)} s`);
    if (stats?.decodeTps) meta.push(`${stats.decodeTps.toFixed(1)} tok/s`);
    if (stats?.promptTokens) meta.push(`${stats.promptTokens} prompt tokens`);
    if (state.generating?.stopped) meta.push("stopped");
    ai.querySelector(".meta-line").innerHTML = meta.map((m) => `<span>${escapeHtml(m)}</span>`).join("");
    if (!overview) state.history.push({ question, answer });
  } catch (err) {
    const stopped = err.message === "stopped" || state.generating?.stopped;
    answerEl.innerHTML = stopped
      ? `<div class="notice">Stopped.</div>`
      : `<div class="notice error">${escapeHtml(friendlyError(err.message))}</div>`;
  } finally {
    llmListeners.delete(requestId);
    state.generating = null;
    setComposerBusy(false);
    scrollToBottom();
  }
}

function stopGenerating() {
  if (!state.generating) return;
  state.generating.stopped = true;
  llm?.postMessage({ type: "interrupt" });
  // if we were still waiting for the model to load, release the waiter
  readyWaiters.forEach((w) => w.reject(new Error("stopped")));
  readyWaiters = [];
}

function setComposerBusy(busy) {
  $("send").hidden = busy;
  $("stop").hidden = !busy;
  updateSendEnabled();
}
function updateSendEnabled() {
  $("send").disabled = !$("input").value.trim();
}

function newChat() {
  if (state.generating) stopGenerating();
  state.history = [];
  $("messages").innerHTML = "";
  $("empty").hidden = false;
  closeViewer();
}

// ------------------------------------------------------------------ settings
function renderModelList() {
  const list = $("model-list");
  if (!caps.webgpu) {
    list.innerHTML = `<label class="model-opt"><input type="radio" name="model" checked disabled />
      <span class="name">${CPU_MODEL.label}</span><span class="desc">${CPU_MODEL.note} ${CPU_MODEL.download}.</span>
      <span class="tag${state.llm.cached[CPU_MODEL.id] ? " cached" : ""}">${state.llm.status === "ready" ? "Loaded" : state.llm.cached[CPU_MODEL.id] ? "Downloaded" : CPU_MODEL.download}</span></label>`;
    return;
  }
  const selected = findGpuModel(resolvedModelId());
  list.innerHTML = GPU_MODELS.map((m) => {
    const id = caps.f16 ? m.id : m.f32;
    const active = state.llm.status === "ready" && state.llm.model === id;
    const cached = state.llm.cached[id];
    const tag = active ? `<span class="tag active">Loaded</span>` : cached ? `<span class="tag cached">Downloaded</span>` : `<span class="tag">${m.download}</span>`;
    return `<label class="model-opt"><input type="radio" name="model" value="${m.id}" ${m === selected ? "checked" : ""} />
      <span class="name">${m.label}</span><span class="desc">${m.note} Needs ~${m.vram} of GPU memory.</span>${tag}</label>`;
  }).join("");
}

function renderSettings() {
  $("device-line").textContent = caps.webgpu
    ? `WebGPU available${caps.adapter ? ` (${caps.adapter})` : ""}${caps.f16 ? "" : ", without 16-bit float support, so f32 builds are used"}.`
    : `WebGPU is not available in this browser, so the chat model runs on the CPU with WebAssembly (${window.crossOriginIsolated ? "multi-threaded" : "single-threaded"}). Chrome, Edge or Safari 26+ on a recent device enables the faster GPU models.`;
  renderModelList();
  $("embedding-select").innerHTML = EMBEDDING_MODELS.map(
    (m) => `<option value="${m.id}" ${m.id === settings.embeddingModel ? "selected" : ""}>${m.label}</option>`,
  ).join("");
  const em = findEmbeddingModel(settings.embeddingModel);
  $("embedding-note").textContent = `${em.note} Download ${em.download}. Changing it re-indexes your library.`;
  $("topk").value = settings.topK;
  $("temperature").value = settings.temperature;
  $("maxtokens").value = settings.maxTokens;
  $("topk-out").textContent = settings.topK;
  $("temp-out").textContent = Number(settings.temperature).toFixed(2);
  $("maxtok-out").textContent = `${settings.maxTokens} tokens`;
  document.querySelectorAll('input[name="theme"]').forEach((r) => (r.checked = r.value === settings.theme));
  updateStorageLine();
}

async function updateStorageLine() {
  try {
    const est = await navigator.storage?.estimate?.();
    const persisted = await navigator.storage?.persisted?.();
    if (est) $("storage-line").textContent = `Using ${fmtMB(est.usage || 0)} of browser storage for models, PDFs and the index${persisted ? " (persistent)" : ""}.`;
  } catch {}
}

async function clearModels() {
  if (!confirm("Delete all downloaded model files? They will be downloaded again next time.")) return;
  const names = await caches.keys();
  await Promise.all(names.filter((n) => /webllm|transformers/i.test(n)).map((n) => caches.delete(n)));
  state.llm.cached = {};
  toast("Downloaded models deleted. Reload the page to free GPU memory.");
  renderModelList();
  updateStorageLine();
}

// --------------------------------------------------------------- sidebar (mobile)
function openSidebar() {
  $("sidebar").classList.add("open");
  $("scrim").hidden = false;
}
function closeSidebarOnMobile() {
  $("sidebar").classList.remove("open");
  $("scrim").hidden = true;
}

// -------------------------------------------------------------------- events
function wireEvents() {
  const input = $("input");
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
    updateSendEnabled();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || state.generating) return;
    input.value = "";
    input.style.height = "auto";
    updateSendEnabled();
    ask(q);
  });
  $("stop").addEventListener("click", stopGenerating);
  $("suggestions").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.overview) ask("", { overview: true });
    else ask(b.dataset.q);
  });

  // upload
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => {
    addFiles(e.target.files);
    e.target.value = "";
  });
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      $("file-input").click();
    }
  });
  for (const t of [dz, document.body]) {
    t.addEventListener("dragover", (e) => {
      if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
      e.preventDefault();
      dz.classList.add("drag");
    });
  }
  document.body.addEventListener("dragleave", (e) => {
    if (!e.relatedTarget) dz.classList.remove("drag");
  });
  document.body.addEventListener("drop", (e) => {
    if (!e.dataTransfer?.files?.length) return;
    e.preventDefault();
    dz.classList.remove("drag");
    addFiles(e.dataTransfer.files);
  });

  // library actions
  $("library").addEventListener("click", (e) => {
    const rm = e.target.closest("[data-remove]");
    if (rm) {
      const doc = state.docs.find((d) => d.id === rm.dataset.remove);
      if (doc && confirm(`Remove “${doc.name}” and its index from this device?`)) rag.postMessage({ type: "remove", docId: doc.id });
    }
    const cancel = e.target.closest("[data-cancel]");
    if (cancel) rag.postMessage({ type: "cancel", jobId: cancel.dataset.cancel });
  });

  // citations and sources
  $("messages").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-source]");
    if (!btn) return;
    const msg = btn.closest(".msg-ai");
    const sources = msgSources.get(msg);
    const n = Number(btn.dataset.source);
    const s = sources?.[n - 1];
    if (!s) return;
    activeCite?.forEach((el) => el.classList.remove("active"));
    activeCite = [...msg.querySelectorAll(`[data-source="${n}"]`)];
    activeCite.forEach((el) => el.classList.add("active"));
    openSource(s).catch((err) => toast(`Could not open the page: ${err.message}`, "error"));
  });
  document.addEventListener("viewer-closed", () => {
    activeCite?.forEach((el) => el.classList.remove("active"));
    activeCite = null;
  });
  $("viewer-close").addEventListener("click", closeViewer);
  $("viewer-prev").addEventListener("click", () => viewerStep(-1));
  $("viewer-next").addEventListener("click", () => viewerStep(1));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && viewerOpen() && !$("settings").open) closeViewer();
  });

  // top bar
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebarOnMobile);
  $("scrim").addEventListener("click", closeSidebarOnMobile);
  $("new-chat").addEventListener("click", newChat);
  $("theme-toggle").addEventListener("click", () => {
    settings.theme = effectiveDark() ? "light" : "dark";
    saveSettings();
    applyTheme();
  });
  darkQuery.addEventListener("change", applyTheme);
  const openSettings = () => {
    renderSettings();
    $("settings").showModal();
  };
  $("open-settings").addEventListener("click", openSettings);
  $("model-chip").addEventListener("click", openSettings);
  $("load-model-cta").addEventListener("click", () => loadModel());
  $("load-samples").addEventListener("click", async () => {
    const names = ["edge-vision-benchmark.pdf", "rag-clinical-systematic-review.pdf"];
    try {
      const files = await Promise.all(
        names.map(async (n) => {
          const res = await fetch(new URL(`../samples/${n}`, import.meta.url));
          if (!res.ok) throw new Error(res.statusText);
          return new File([await res.blob()], n, { type: "application/pdf" });
        }),
      );
      addFiles(files);
      toast("Loaded two fictional sample papers (made for this demo).");
    } catch (err) {
      toast(`Could not load the samples: ${err.message}`, "error");
    }
  });

  // settings form
  $("model-list").addEventListener("change", (e) => {
    if (e.target.name !== "model") return;
    settings.gpuModel = e.target.value;
    saveSettings();
    renderModelState();
  });
  $("load-model").addEventListener("click", () => {
    loadModel();
    $("settings").close();
  });
  $("embedding-select").addEventListener("change", (e) => {
    settings.embeddingModel = e.target.value;
    saveSettings();
    state.embed = { status: "loading", text: "Switching embedding model…" };
    renderEmbedStatus();
    rag.postMessage({ type: "set-embedding", embeddingModel: settings.embeddingModel });
    renderSettings();
  });
  const bindRange = (id, key, fmt, parse = Number) =>
    $(id).addEventListener("input", (e) => {
      settings[key] = parse(e.target.value);
      $(`${id === "maxtokens" ? "maxtok" : id === "temperature" ? "temp" : id}-out`).textContent = fmt(settings[key]);
      saveSettings();
    });
  bindRange("topk", "topK", (v) => v);
  bindRange("temperature", "temperature", (v) => v.toFixed(2));
  bindRange("maxtokens", "maxTokens", (v) => `${v} tokens`);
  document.querySelectorAll('input[name="theme"]').forEach((r) =>
    r.addEventListener("change", () => {
      settings.theme = r.value;
      saveSettings();
      applyTheme();
    }),
  );
  $("clear-library").addEventListener("click", () => {
    if (confirm("Remove every document and its index from this device?")) {
      rag.postMessage({ type: "clear" });
      newChat();
    }
  });
  $("clear-models").addEventListener("click", clearModels);
  $("settings").addEventListener("click", (e) => {
    if (e.target === $("settings")) $("settings").close(); // click on backdrop
  });
}

// ---------------------------------------------------------------------- boot
async function detectGpu() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      caps.webgpu = true;
      caps.f16 = adapter.features.has("shader-f16");
      caps.adapter = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(" ");
    }
  } catch {}
}

/**
 * Without WebGPU the chat model runs on the CPU, where multi-threaded WebAssembly is
 * several times faster but needs cross-origin isolation. GitHub Pages cannot set the
 * headers, so a tiny service worker adds them, followed by one reload.
 */
async function enableCrossOriginIsolation() {
  if (window.crossOriginIsolated || !("serviceWorker" in navigator)) return false;
  try {
    await navigator.serviceWorker.register(new URL("../coi-sw.js", import.meta.url));
    await navigator.serviceWorker.ready;
    if (sessionStorage.getItem("synapse.coi-reload")) return false; // tried once: carry on single-threaded
    sessionStorage.setItem("synapse.coi-reload", "1");
    location.reload();
    return true;
  } catch (err) {
    console.warn("Cross-origin isolation unavailable, CPU inference stays single-threaded", err);
    return false;
  }
}

async function boot() {
  applyTheme();
  wireEvents();
  renderLibrary();
  renderEmbedStatus();
  updateSendEnabled();
  navigator.storage?.persist?.().catch(() => {});
  await detectGpu();
  if (!caps.webgpu && (await enableCrossOriginIsolation())) return; // reloading once
  startLlmWorker();
  renderModelState();
  // Reload with a cached model: load it straight away (no download needed).
  if (await checkModelCache()) loadModel();
  else renderSteps();
}

boot();

// Exposed for debugging in the console only.
window.synapse = { state, settings, caps, search: (query, topK = 6) => ragRequest({ type: "search", query, topK }) };
