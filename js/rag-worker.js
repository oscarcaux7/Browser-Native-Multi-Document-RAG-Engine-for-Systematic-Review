// RAG worker: PDF parsing, chunking, embeddings, hybrid search and index
// persistence all run here, so the page stays responsive while indexing.
import { LIBS, CHUNKING, findEmbeddingModel } from "./config.js";
import { chunkDocument, pageItemsToText } from "./chunker.js";
import { BM25, denseSearch, rrf } from "./retrieval.js";
import { store } from "./store.js";

const { AutoTokenizer, AutoModel, env } = await import(LIBS.transformers);
env.allowLocalModels = false; // models come from the Hugging Face Hub, cached by the browser

let pdfjs = null;
async function getPdfjs() {
  if (!pdfjs) {
    pdfjs = await import(LIBS.pdfjs);
    pdfjs.GlobalWorkerOptions.workerSrc = LIBS.pdfjsWorker;
  }
  return pdfjs;
}

const post = (msg) => self.postMessage(msg);

// ---------------------------------------------------------------- embeddings
const embedder = { spec: null, tokenizer: null, model: null, device: null };

async function detectDevice() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) return { device: "webgpu", f16: adapter.features.has("shader-f16") };
  } catch {}
  return { device: "wasm", f16: false };
}

async function loadEmbedder(modelId) {
  const spec = findEmbeddingModel(modelId);
  const { device, f16 } = await detectDevice();
  let dtype = device === "webgpu" ? (f16 ? spec.dtype.webgpu : spec.dtype.webgpuNoF16) : spec.dtype.wasm;
  const progress_callback = (p) => {
    if (p.status === "progress_total") post({ type: "model-progress", kind: "embedding", loaded: p.loaded, total: p.total });
  };
  post({ type: "model-progress", kind: "embedding", loaded: 0, total: 0, label: spec.label });
  await embedder.model?.dispose?.();
  const tokenizer = await AutoTokenizer.from_pretrained(spec.id, { progress_callback });
  let model;
  let used = device;
  try {
    model = await AutoModel.from_pretrained(spec.id, { device, dtype, progress_callback });
  } catch (err) {
    if (device !== "webgpu") throw err;
    console.warn("WebGPU embedding failed, falling back to WASM", err);
    used = "wasm";
    dtype = spec.dtype.wasm;
    model = await AutoModel.from_pretrained(spec.id, { device: used, dtype, progress_callback });
  }
  Object.assign(embedder, { spec, tokenizer, model, device: used });
  return { device: used, dtype };
}

/** Embed a batch of strings into L2-normalised vectors (one Float32Array, row-major). */
async function embed(texts) {
  const { spec, tokenizer, model } = embedder;
  const inputs = await tokenizer(texts, { padding: true, truncation: true, max_length: spec.maxTokens });
  const out = await model(inputs);
  const dims = spec.dims;
  const result = new Float32Array(texts.length * dims);
  if (spec.pooling === "sentence_embedding") {
    result.set(out.sentence_embedding.data.subarray(0, texts.length * dims));
  } else {
    const h = out.last_hidden_state; // [batch, tokens, dims]
    const [, T, D] = h.dims;
    for (let b = 0; b < texts.length; b++) result.set(h.data.subarray(b * T * D, b * T * D + dims), b * dims); // CLS
  }
  for (let b = 0; b < texts.length; b++) {
    let n = 0;
    for (let i = 0; i < dims; i++) n += result[b * dims + i] ** 2;
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < dims; i++) result[b * dims + i] /= n;
  }
  return result;
}

// ------------------------------------------------------------------ the index
const index = {
  chunks: [], // {id, docId, docName, page, text, ord}
  byId: new Map(),
  matrix: new Float32Array(0),
  bm25: new BM25(),
};
let docs = new Map(); // id -> doc record

function addToIndex(doc, chunks, vectors) {
  const dims = embedder.spec.dims;
  const start = index.chunks.length;
  const grown = new Float32Array(index.matrix.length + vectors.length);
  grown.set(index.matrix);
  grown.set(vectors, index.matrix.length);
  index.matrix = grown;
  chunks.forEach((c, i) => {
    const rec = { id: `${doc.id}:${i}`, docId: doc.id, docName: doc.name, page: c.page, text: c.text, ord: i, row: start + i };
    index.chunks.push(rec);
    index.byId.set(rec.id, rec);
    index.bm25.add(rec.id, `${doc.name} ${c.text}`);
  });
  if (index.matrix.length !== index.chunks.length * dims) throw new Error("index/vector size mismatch");
}

function resetIndex() {
  index.chunks = [];
  index.byId = new Map();
  index.matrix = new Float32Array(0);
  index.bm25 = new BM25();
}

function removeFromIndex(docId) {
  const dims = embedder.spec.dims;
  const keep = index.chunks.filter((c) => c.docId !== docId);
  const m = new Float32Array(keep.length * dims);
  keep.forEach((c, i) => {
    m.set(index.matrix.subarray(c.row * dims, c.row * dims + dims), i * dims);
    c.row = i;
  });
  index.chunks = keep;
  index.matrix = m;
  index.byId = new Map(keep.map((c) => [c.id, c]));
  index.bm25.remove((id) => id.startsWith(`${docId}:`));
}

async function loadIndexFromDb() {
  resetIndex();
  docs = new Map((await store.listDocs()).map((d) => [d.id, d]));
  const saved = await store.indexesForModel(embedder.spec.id);
  for (const rec of saved) {
    const doc = docs.get(rec.docId);
    if (doc) addToIndex(doc, rec.chunks, rec.vectors);
  }
  const indexed = new Set(saved.map((r) => r.docId));
  return [...docs.values()].filter((d) => !indexed.has(d.id)); // docs that need (re)embedding
}

function docSummaries() {
  const counts = new Map();
  for (const c of index.chunks) counts.set(c.docId, (counts.get(c.docId) || 0) + 1);
  return [...docs.values()]
    .sort((a, b) => a.addedAt - b.addedAt)
    .map((d) => ({ id: d.id, name: d.name, size: d.size, pages: d.pages, chunks: counts.get(d.id) || 0 }));
}

// ---------------------------------------------------------------- ingestion
async function hashBytes(buf) {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(digest)].slice(0, 12).map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function extractPages(buf, onPage) {
  const lib = await getPdfjs();
  const task = lib.getDocument({ data: new Uint8Array(buf), isEvalSupported: false });
  const pdf = await task.promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    pages.push(pageItemsToText(content.items));
    page.cleanup();
    onPage(i, pdf.numPages);
  }
  await task.destroy();
  return pages;
}

const BATCH = 16;

async function embedChunks(doc, chunks, report, jobId = doc.id) {
  const dims = embedder.spec.dims;
  const vectors = new Float32Array(chunks.length * dims);
  for (let i = 0; i < chunks.length; i += BATCH) {
    if (cancelled.has(jobId)) throw new Error("cancelled");
    const batch = chunks.slice(i, i + BATCH).map((c) => embedder.spec.docPrefix(doc.name) + c.text);
    vectors.set(await embed(batch), i * dims);
    report(Math.min(i + BATCH, chunks.length), chunks.length);
  }
  return vectors;
}

const cancelled = new Set();

async function ingestFile(file, jobId) {
  const buf = await file.arrayBuffer();
  const id = await hashBytes(buf);
  // Keep a copy for the page viewer now: PDF.js transfers (detaches) the buffer it parses.
  const blob = new Blob([buf], { type: "application/pdf" });
  if (docs.has(id)) {
    post({ type: "ingest-skip", jobId, name: file.name, reason: "already in your library" });
    return;
  }
  const t0 = performance.now();
  const job = { type: "ingest-progress", jobId, name: file.name };
  post({ ...job, phase: "reading", done: 0, total: 1 });
  let pages;
  try {
    pages = await extractPages(buf, (done, total) => post({ ...job, phase: "parsing", done, total }));
  } catch (err) {
    post({ type: "ingest-error", jobId, name: file.name, error: `Could not read this PDF (${err.message})` });
    return;
  }
  const chunks = chunkDocument(pages, CHUNKING);
  if (!chunks.length) {
    post({ type: "ingest-error", jobId, name: file.name, error: "No selectable text found (scanned PDFs need OCR first)." });
    return;
  }
  const doc = { id, name: file.name, size: file.size, pages: pages.length, addedAt: Date.now() };
  try {
    const vectors = await embedChunks(doc, chunks, (done, total) => post({ ...job, phase: "embedding", done, total }), jobId);
    await store.putDocWithFile(doc, blob);
    await store.putIndex(embedder.spec.id, id, chunks, vectors);
    docs.set(id, doc);
    addToIndex(doc, chunks, vectors);
  } catch (err) {
    cancelled.delete(jobId);
    post({ type: "ingest-error", jobId, name: file.name, error: err.message === "cancelled" ? "Cancelled." : err.message });
    return;
  }
  const seconds = (performance.now() - t0) / 1000;
  post({ type: "ingest-done", jobId, name: file.name, pages: pages.length, chunks: chunks.length, seconds });
  post({ type: "library", docs: docSummaries() });
}

/** Re-embed stored PDFs that have no vectors for the current embedding model. */
async function reembed(missing) {
  for (const doc of missing) {
    const blob = await store.getFile(doc.id);
    if (!blob) continue;
    const job = { type: "ingest-progress", jobId: doc.id, name: doc.name };
    const pages = await extractPages(await blob.arrayBuffer(), (done, total) => post({ ...job, phase: "parsing", done, total }));
    const chunks = chunkDocument(pages, CHUNKING);
    const vectors = await embedChunks(doc, chunks, (done, total) => post({ ...job, phase: "embedding", done, total }));
    await store.putIndex(embedder.spec.id, doc.id, chunks, vectors);
    addToIndex(doc, chunks, vectors);
    post({ type: "ingest-done", jobId: doc.id, name: doc.name, pages: pages.length, chunks: chunks.length, reindexed: true });
  }
  post({ type: "library", docs: docSummaries() });
}

// ------------------------------------------------------------------- search
async function search(query, topK) {
  if (!index.chunks.length) return [];
  const qv = await embed([embedder.spec.queryPrefix + query]);
  const ids = index.chunks.map((c) => c.id);
  const dense = denseSearch(index.matrix, embedder.spec.dims, ids, qv, 40);
  const sparse = index.bm25.search(query, 40);
  const denseScore = new Map(dense.map((d) => [d.id, d.score]));
  return rrf([dense, sparse])
    .slice(0, topK)
    .map(({ id, score }) => {
      const c = index.byId.get(id);
      return { id, docId: c.docId, docName: c.docName, page: c.page, text: c.text, score, similarity: denseScore.get(id) ?? null };
    });
}

/** Representative passages for a whole-library overview: the opening chunks of each document. */
function overviewSources(perDocChars, totalChars = 5600) {
  const out = [];
  let total = 0;
  for (const doc of docs.values()) {
    if (total >= totalChars) break; // stay inside a 4k-token context window
    let used = 0;
    for (const c of index.chunks) {
      if (c.docId !== doc.id) continue;
      if (used + c.text.length > perDocChars && used > 0) break;
      if (total + c.text.length > totalChars) break;
      out.push({ id: c.id, docId: c.docId, docName: c.docName, page: c.page, text: c.text, score: null });
      used += c.text.length;
      total += c.text.length;
    }
  }
  return out;
}

// ------------------------------------------------------------------ messages
let queue = Promise.resolve(); // serialise indexing jobs; searches run between batches
/** Queue a job; a failure is reported but never blocks the jobs after it. */
function enqueue(job, during = "job") {
  queue = queue.then(job).catch((err) => {
    console.error(err);
    post({ type: "error", during, error: err?.message || String(err) });
  });
}

async function switchEmbedder(modelId) {
  try {
    const info = await loadEmbedder(modelId);
    const missing = await loadIndexFromDb();
    post({ type: "ready", ...info, model: embedder.spec.id, docs: docSummaries() });
    await reembed(missing);
  } catch (err) {
    console.error(err);
    post({ type: "error", during: "embedder", model: modelId, error: err?.message || String(err) });
  }
}

self.onmessage = async ({ data: msg }) => {
  try {
    switch (msg.type) {
      case "init":
      case "set-embedding":
        enqueue(() => switchEmbedder(msg.embeddingModel), "embedder");
        break;
      case "ingest":
        for (const { file, jobId } of msg.files) {
          post({ type: "ingest-progress", jobId, name: file.name, phase: "queued", done: 0, total: 0 });
          enqueue(() => (cancelled.delete(jobId) ? post({ type: "ingest-error", jobId, name: file.name, error: "Cancelled." }) : ingestFile(file, jobId)), "ingest");
        }
        break;
      case "cancel":
        cancelled.add(msg.jobId);
        break;
      case "search": {
        const sources = msg.mode === "overview" ? overviewSources(msg.perDocChars) : await search(msg.query, msg.topK);
        post({ type: "results", requestId: msg.requestId, sources });
        break;
      }
      case "remove":
        await store.removeDoc(msg.docId);
        docs.delete(msg.docId);
        removeFromIndex(msg.docId);
        post({ type: "library", docs: docSummaries() });
        break;
      case "clear":
        await store.clearAll();
        docs.clear();
        resetIndex();
        post({ type: "library", docs: [] });
        break;
    }
  } catch (err) {
    console.error(err);
    post({ type: "error", requestId: msg.requestId, error: err.message || String(err) });
  }
};

post({ type: "booted" });
