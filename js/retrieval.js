// Hybrid retrieval: dense vectors (cosine) + BM25 keywords, merged with
// Reciprocal Rank Fusion. Pure functions; the RAG worker owns the data.

const STOP = new Set(
  ("a an and are as at be but by for from has have in is it its of on or that the this to was were " +
    "will with what which who whom how why when where do does did can could should would about into " +
    "than then there these those their they them we our you your i me my not no so if also may").split(" "),
);

export function tokenize(text) {
  return (text.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").match(/[\p{L}\p{N}]+/gu) || [])
    .filter((t) => t.length > 1 && !STOP.has(t));
}

/** Incrementally maintained BM25 index. */
export class BM25 {
  constructor({ k1 = 1.2, b = 0.75 } = {}) {
    this.k1 = k1;
    this.b = b;
    this.docs = []; // {id, tf: Map, len}
    this.df = new Map();
    this.totalLen = 0;
  }

  add(id, text) {
    const toks = tokenize(text);
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    for (const t of tf.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
    this.docs.push({ id, tf, len: toks.length });
    this.totalLen += toks.length;
  }

  remove(predicate) {
    const keep = [];
    for (const d of this.docs) {
      if (predicate(d.id)) {
        for (const t of d.tf.keys()) {
          const n = this.df.get(t) - 1;
          if (n > 0) this.df.set(t, n);
          else this.df.delete(t);
        }
        this.totalLen -= d.len;
      } else keep.push(d);
    }
    this.docs = keep;
  }

  /** @returns {{id: string, score: number}[]} sorted, best first */
  search(query, limit = 20) {
    const q = [...new Set(tokenize(query))];
    const N = this.docs.length;
    if (!N || !q.length) return [];
    const avg = this.totalLen / N || 1;
    const scored = [];
    for (const d of this.docs) {
      let s = 0;
      for (const t of q) {
        const f = d.tf.get(t);
        if (!f) continue;
        const df = this.df.get(t) || 0;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        s += idf * ((f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * d.len) / avg)));
      }
      if (s > 0) scored.push({ id: d.id, score: s });
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, limit);
  }
}

/**
 * Dense search over L2-normalised vectors stored row-major in one Float32Array.
 * Dot product equals cosine similarity for normalised vectors.
 */
export function denseSearch(matrix, dims, ids, query, limit = 20) {
  const n = ids.length;
  const scores = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    let s = 0;
    const off = r * dims;
    for (let c = 0; c < dims; c++) s += matrix[off + c] * query[c];
    scores[r] = s;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  return order.slice(0, limit).map((i) => ({ id: ids[i], score: scores[i] }));
}

/** Reciprocal Rank Fusion of several ranked lists. */
export function rrf(lists, { k = 60, weights } = {}) {
  const fused = new Map();
  lists.forEach((list, li) => {
    const w = weights ? weights[li] : 1;
    list.forEach((item, rank) => {
      fused.set(item.id, (fused.get(item.id) || 0) + w / (k + rank + 1));
    });
  });
  return [...fused.entries()].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score);
}
