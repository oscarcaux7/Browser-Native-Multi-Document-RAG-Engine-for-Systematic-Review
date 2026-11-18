// Text clean-up and page-aware chunking. Pure functions, no DOM: used by the
// RAG worker and by the Node unit tests.

/**
 * Turn PDF.js text items for one page into readable text.
 * Keeps line breaks where PDF.js reports them, re-joins words hyphenated
 * across lines and collapses runs of whitespace.
 * @param {{str: string, hasEOL?: boolean}[]} items
 */
export function pageItemsToText(items) {
  let out = "";
  for (const it of items) {
    if (typeof it.str !== "string") continue;
    out += it.str;
    out += it.hasEOL ? "\n" : "";
  }
  return cleanText(out);
}

export function cleanText(text) {
  return text
    .replace(/­/g, "") // soft hyphens
    .replace(/[ \t\f\v ]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/([a-z])-\n([a-z])/g, "$1$2") // "retrie-\nval" -> "retrieval"
    .replace(/\n{2,}/g, "\n\n")
    .replace(/(?<!\n)\n(?!\n)/g, " ") // single newlines are soft wraps
    .replace(/ {2,}/g, " ")
    .trim();
}

const ABBREV = /\b(?:e\.g|i\.e|et al|Fig|fig|Eq|eq|Tab|vs|cf|approx|resp|No|Dr|Mr|Ms|Prof|Sec|Ref)\.$/;

/** Split text into sentences, tolerant of common scientific abbreviations. */
export function splitSentences(text) {
  const parts = text.split(/(?<=[.!?…])\s+(?=["“'(\[]?[A-Z0-9])|\n\n+/);
  const sentences = [];
  let buf = "";
  for (const p of parts) {
    if (!p) continue;
    buf = buf ? `${buf} ${p}` : p;
    if (ABBREV.test(buf)) continue; // keep "et al." attached to what follows
    sentences.push(buf.trim());
    buf = "";
  }
  if (buf.trim()) sentences.push(buf.trim());
  return sentences.filter(Boolean);
}

/** Hard-wrap a string that is longer than `max` on word boundaries. */
function hardWrap(s, max) {
  if (s.length <= max) return [s];
  const out = [];
  let rest = s;
  while (rest.length > max) {
    let cut = rest.lastIndexOf(" ", max);
    if (cut < max * 0.5) cut = max;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) out.push(rest);
  return out;
}

/**
 * Chunk one page of text into overlapping, sentence-aligned windows.
 * Chunks never cross a page boundary, so every chunk has an exact page to cite.
 * @returns {string[]}
 */
export function chunkPage(text, { targetChars = 900, overlapChars = 180, minChars = 40 } = {}) {
  const sentences = splitSentences(text).flatMap((s) => hardWrap(s, targetChars));
  const chunks = [];
  let current = [];
  let len = 0;

  const flush = () => {
    const chunk = current.join(" ").trim();
    if (chunk.length >= minChars) chunks.push(chunk);
    // carry trailing sentences forward as overlap
    const carry = [];
    let carryLen = 0;
    for (let i = current.length - 1; i >= 0; i--) {
      if (carryLen + current[i].length > overlapChars) break;
      carry.unshift(current[i]);
      carryLen += current[i].length + 1;
    }
    current = carry;
    len = carryLen;
  };

  for (const s of sentences) {
    if (len + s.length > targetChars && len > 0) flush();
    current.push(s);
    len += s.length + 1;
  }
  // last chunk: only emit if it adds something beyond the overlap
  const tail = current.join(" ").trim();
  const last = chunks[chunks.length - 1];
  if (tail.length >= minChars && (!last || !last.endsWith(tail))) chunks.push(tail);
  return chunks;
}

/**
 * Chunk a whole document given per-page text.
 * @param {string[]} pages text of page 1..n
 * @returns {{page: number, text: string}[]}
 */
export function chunkDocument(pages, opts) {
  const out = [];
  pages.forEach((text, i) => {
    for (const t of chunkPage(text, opts)) out.push({ page: i + 1, text: t });
  });
  return out;
}
