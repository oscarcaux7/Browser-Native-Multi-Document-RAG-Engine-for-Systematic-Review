// Prompt construction and citation parsing. Pure functions (unit-tested).

export const SYSTEM_PROMPT = `You are Synapse, a careful research assistant.
Answer using ONLY the sources provided by the user (each one is wrapped in a <source id="n"> tag).
Write the answer in your own words, in full sentences. Never copy the source tags or repeat a sentence.
After every claim, cite the supporting source id in square brackets, for example [2] or [1][3].
If the sources do not contain the answer, say that you could not find it in the documents. Do not invent facts, numbers or references.
Be concise: short paragraphs or bullet points. Answer in the language of the question.`;

/** Sources as tagged blocks. Tags (rather than "[1] title" headers) stop small models from echoing them. */
export function formatSources(sources) {
  return sources
    .map((s, i) => `<source id="${i + 1}" document="${s.docName.replace(/"/g, "'")}" page="${s.page}">\n${s.text}\n</source>`)
    .join("\n");
}

/**
 * Does this question refer back to the previous turn ("what about its power draw?")?
 * Only then is the last exchange sent along: small models otherwise reuse the previous
 * answer for unrelated questions.
 */
export function isFollowUp(question, history = []) {
  if (!history.length) return false;
  const q = question.trim().toLowerCase();
  if (q.split(/\s+/).length <= 4) return true;
  if (/^(and|but|so|also|what about|how about|why|then)\b/.test(q)) return true;
  return /\b(it|its|they|them|their|these|those|this one|that one|the same|above|previous|earlier|you said|your answer)\b/.test(q);
}

/**
 * Build chat messages for one RAG turn.
 * @param {object} p
 * @param {string} p.question
 * @param {{docName: string, page: number, text: string}[]} p.sources
 * @param {{question: string, answer: string}[]} [p.history] previous turns (most recent last)
 * @param {"qa"|"overview"} [p.mode]
 */
export function buildMessages({ question, sources, history = [], mode = "qa", lowRelevance = false }) {
  const messages = [{ role: "system", content: SYSTEM_PROMPT }];
  // Keep only the last exchange, trimmed, so follow-up questions have context
  // without crowding the 4k-token window of small models.
  const last = mode === "qa" && isFollowUp(question, history) ? history[history.length - 1] : null;
  if (last) {
    messages.push({ role: "user", content: last.question });
    messages.push({ role: "assistant", content: stripCitations(last.answer).slice(0, 600) });
  }
  const task =
    mode === "overview"
      ? `Write a short structured overview of these documents for a literature review: for each document give its topic, method and key findings, then note agreements or differences between them. Cite sources as [n].`
      : `Question: ${question}\n${
          lowRelevance
            ? "None of these sources looks closely related to the question. If they do not answer it, reply that the documents do not cover it."
            : "Answer from the sources above and cite them like [1]."
        }`;
  messages.push({
    role: "user",
    content: sources.length
      ? `${formatSources(sources)}\n\n${task}`
      : `No sources were retrieved from the documents.\n\n${task}`,
  });
  return messages;
}

/**
 * Clean a (possibly partial) model answer for display: drop a reasoning block
 * (<think>…</think>) and any tag-like markup small models copy from the
 * <source> blocks (e.g. "</source>", "<no_source>").
 */
export function stripThinking(text) {
  let t = text.replace(/<think>[\s\S]*?<\/think>\s*/g, "");
  const open = t.indexOf("<think>");
  if (open !== -1) t = t.slice(0, open); // still thinking: hide partial block
  t = t.replace(/<\/?[a-z_][\w-]*(\s[^<>]*)?>/gi, "");
  return t.trimStart();
}

export function stripCitations(text) {
  return text.replace(CITE_RE, "").replace(/ {2,}/g, " ").replace(/ +([.,;:!?])/g, "$1").trim();
}

// [2]  [1, 3]  [1-3]  [Source 2]  [sources 1 and 4]
export const CITE_RE = /\[(?:sources?\s*)?(\d+(?:\s*(?:,|;|and|–|-)\s*\d+)*)\]/gi;

/** Expand the inside of a citation bracket to source numbers. */
export function expandCitation(inner) {
  const nums = [];
  for (const part of inner.split(/\s*(?:,|;|and)\s*/i)) {
    const range = part.match(/^(\d+)\s*[–-]\s*(\d+)$/);
    if (range) {
      const [a, b] = [Number(range[1]), Number(range[2])];
      if (b >= a && b - a < 20) for (let i = a; i <= b; i++) nums.push(i);
    } else if (/^\d+$/.test(part.trim())) nums.push(Number(part));
  }
  return nums;
}

/** Ordered, de-duplicated list of valid source numbers cited in `text`. */
export function citedSources(text, count) {
  const seen = new Set();
  for (const m of text.matchAll(CITE_RE)) {
    for (const n of expandCitation(m[1])) if (n >= 1 && n <= count) seen.add(n);
  }
  return [...seen];
}
