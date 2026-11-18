// Source viewer: renders the cited PDF page from the local copy (IndexedDB)
// and highlights the retrieved passage using PDF.js's text layer.
import { LIBS } from "./config.js";
import { store } from "./store.js";

const $ = (id) => document.getElementById(id);
let pdfjs = null;
const current = { docId: null, task: null, pdf: null, page: 1, quote: "", token: 0 };

async function lib() {
  if (!pdfjs) {
    pdfjs = await import(LIBS.pdfjs);
    pdfjs.GlobalWorkerOptions.workerSrc = LIBS.pdfjsWorker;
  }
  return pdfjs;
}

const norm = (s) => s.toLowerCase().normalize("NFKD").replace(/[^\p{L}\p{N}]+/gu, "");

export function isOpen() {
  return !$("viewer").hidden;
}

export function closeViewer() {
  $("viewer").hidden = true;
  document.getElementById("app").classList.remove("with-viewer");
  document.dispatchEvent(new CustomEvent("viewer-closed"));
}

/** @param {{docId: string, docName: string, page: number, text: string}} source */
export async function openSource(source) {
  $("viewer").hidden = false;
  document.getElementById("app").classList.add("with-viewer");
  $("viewer-doc").textContent = source.docName;
  $("viewer-quote").textContent = source.text;
  current.quote = source.text;
  if (current.docId !== source.docId) {
    await current.task?.destroy();
    current.pdf = null;
    current.docId = source.docId;
    const blob = await store.getFile(source.docId);
    if (!blob) {
      $("page-wrap").innerHTML = `<div class="loading">This document is no longer in your library.</div>`;
      return;
    }
    const { getDocument } = await lib();
    current.task = getDocument({ data: new Uint8Array(await blob.arrayBuffer()) });
    current.pdf = await current.task.promise;
  }
  await renderPage(source.page, true);
}

export async function step(delta) {
  if (!current.pdf) return;
  const next = Math.min(Math.max(current.page + delta, 1), current.pdf.numPages);
  if (next !== current.page) await renderPage(next, false);
}

async function renderPage(n, highlight) {
  const token = ++current.token;
  current.page = n;
  const pdf = current.pdf;
  $("viewer-page").textContent = `Page ${n} of ${pdf.numPages}`;
  $("viewer-prev").disabled = n <= 1;
  $("viewer-next").disabled = n >= pdf.numPages;
  $("viewer-quote").hidden = !highlight;

  const wrap = $("page-wrap");
  const page = await pdf.getPage(n);
  if (token !== current.token) return;
  const width = Math.max(240, $("viewer-body").clientWidth - 32);
  const base = page.getViewport({ scale: 1 });
  const viewport = page.getViewport({ scale: width / base.width });
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);

  const canvas = document.createElement("canvas");
  canvas.width = Math.floor(viewport.width * dpr);
  canvas.height = Math.floor(viewport.height * dpr);
  const layer = document.createElement("div");
  layer.className = "textLayer";

  wrap.style.width = `${viewport.width}px`;
  wrap.style.setProperty("--total-scale-factor", String(viewport.scale));
  await page.render({ canvas, canvasContext: canvas.getContext("2d"), viewport, transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : null }).promise;
  if (token !== current.token) return;
  wrap.replaceChildren(canvas, layer);

  const { TextLayer } = await lib();
  const tl = new TextLayer({ textContentSource: page.streamTextContent(), container: layer, viewport });
  await tl.render();
  if (token !== current.token || !highlight) return;

  // Highlight text-layer spans that belong to the retrieved passage.
  const target = norm(current.quote);
  let first = null;
  for (const span of layer.querySelectorAll("span")) {
    const t = norm(span.textContent || "");
    if (t.length >= 4 && target.includes(t)) {
      span.classList.add("hit");
      first ??= span;
    }
  }
  if (first) first.scrollIntoView({ block: "center", behavior: "smooth" });
  else $("viewer-body").scrollTop = 0;
}
