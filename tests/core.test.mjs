// Unit tests for the pure modules. Run with: node --test tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { cleanText, splitSentences, chunkPage, chunkDocument, pageItemsToText } from "../js/chunker.js";
import { BM25, denseSearch, rrf, tokenize } from "../js/retrieval.js";
import { buildMessages, citedSources, expandCitation, stripThinking, isFollowUp } from "../js/prompt.js";
import { renderMarkdown } from "../js/markdown.js";

test("cleanText joins hyphenated line breaks and soft wraps", () => {
  assert.equal(cleanText("retrie-\nval augmented\ngeneration"), "retrieval augmented generation");
  assert.equal(cleanText("a   b\t c"), "a b c");
});

test("pageItemsToText respects hasEOL", () => {
  const t = pageItemsToText([{ str: "Hello", hasEOL: true }, { str: "world." }]);
  assert.equal(t, "Hello world.");
});

test("splitSentences keeps scientific abbreviations attached", () => {
  const s = splitSentences("Smith et al. Reported gains. See Fig. 3 for details. Done!");
  assert.deepEqual(s, ["Smith et al. Reported gains.", "See Fig. 3 for details.", "Done!"]);
});

test("chunkPage respects target size and overlaps", () => {
  const text = Array.from({ length: 40 }, (_, i) => `Sentence number ${i} talks about topic ${i}.`).join(" ");
  const chunks = chunkPage(text, { targetChars: 200, overlapChars: 60, minChars: 10 });
  assert.ok(chunks.length > 5);
  for (const c of chunks) assert.ok(c.length <= 260, `chunk too long: ${c.length}`);
  // consecutive chunks share the overlapping sentence
  const lastSentence = chunks[0].split(". ").pop();
  assert.ok(chunks[1].includes(lastSentence.replace(/\.$/, "")));
});

test("chunkDocument keeps exact page numbers", () => {
  const chunks = chunkDocument(["Page one has enough text to be kept as a chunk.", "", "Third page text is long enough too."], {
    targetChars: 900,
    overlapChars: 100,
    minChars: 20,
  });
  assert.deepEqual(chunks.map((c) => c.page), [1, 3]);
});

test("BM25 ranks exact keyword matches first and supports removal", () => {
  const bm = new BM25();
  bm.add("a", "The Jetson Orin Nano reached 31 FPS with INT8 quantization.");
  bm.add("b", "We review retrieval augmented generation for clinical decision support.");
  bm.add("c", "Quantization reduces memory use on embedded devices.");
  assert.equal(bm.search("Jetson FPS")[0].id, "a");
  bm.remove((id) => id === "a");
  assert.equal(bm.search("Jetson")[0], undefined);
  assert.equal(bm.search("quantization")[0].id, "c");
});

test("denseSearch + rrf fuse rankings", () => {
  const m = new Float32Array([1, 0, 0, 1, 0.7071, 0.7071]);
  const dense = denseSearch(m, 2, ["x", "y", "z"], [1, 0], 3);
  assert.deepEqual(dense.map((d) => d.id), ["x", "z", "y"]);
  const fused = rrf([dense, [{ id: "z" }, { id: "y" }]]);
  assert.equal(fused[0].id, "z"); // 2nd + 1st beats 1st + absent
  assert.equal(fused[2].id, "x"); // only in one list
});

test("tokenize drops stop words and accents", () => {
  assert.deepEqual(tokenize("The Évaluation of a model"), ["evaluation", "model"]);
});

test("citation parsing handles lists, ranges and bounds", () => {
  assert.deepEqual(expandCitation("1, 3"), [1, 3]);
  assert.deepEqual(expandCitation("2-4"), [2, 3, 4]);
  assert.deepEqual(citedSources("A [2]. B [1][2]. C [Source 3]. D [9].", 4), [2, 1, 3]);
});

test("stripThinking hides reasoning blocks, even unfinished ones", () => {
  assert.equal(stripThinking("<think>hmm</think>\n\nAnswer"), "Answer");
  assert.equal(stripThinking("<think>still going"), "");
  assert.equal(stripThinking("Not covered.<no_source>No sources.</no_source> x < 5"), "Not covered.No sources. x < 5");
});

test("isFollowUp only flags questions that refer back", () => {
  const h = [{ question: "q", answer: "a" }];
  assert.equal(isFollowUp("What is its power draw?", h), true);
  assert.equal(isFollowUp("And on the Pi?", h), true);
  assert.equal(isFollowUp("What is the boiling point of mercury according to these documents?", []), false);
  assert.equal(isFollowUp("What limitations do the authors mention in the benchmark paper?", h), false);
});

test("unrelated questions do not carry the previous exchange", () => {
  const msgs = buildMessages({
    question: "Which datasets were used for calibration in the study?",
    sources: [],
    history: [{ question: "prev", answer: "prev answer" }],
  });
  assert.equal(msgs.length, 2);
});

test("low-relevance retrieval tells the model it may not be covered", () => {
  const msgs = buildMessages({ question: "Boiling point of mercury?", sources: [{ docName: "a.pdf", page: 1, text: "x" }], lowRelevance: true });
  assert.match(msgs.at(-1).content, /do not cover it/);
});

test("buildMessages includes tagged sources and, for follow-ups, the last turn only", () => {
  const msgs = buildMessages({
    question: "What FPS?",
    sources: [{ docName: "a.pdf", page: 4, text: "31 FPS" }],
    history: [
      { question: "old", answer: "old answer" },
      { question: "prev", answer: "prev answer [1]" },
    ],
  });
  assert.equal(msgs.length, 4);
  assert.equal(msgs[1].content, "prev");
  assert.equal(msgs[2].content, "prev answer");
  assert.match(msgs[3].content, /<source id="1" document="a\.pdf" page="4">\n31 FPS\n<\/source>/);
  assert.match(msgs[3].content, /Question: What FPS\?\n/);
});

test("renderMarkdown escapes HTML and renders citations as buttons", () => {
  const html = renderMarkdown("**Bold** <script>x</script> claim [1][7]\n\n- item [2]", 2);
  assert.ok(html.includes("<strong>Bold</strong>"));
  assert.ok(html.includes("&lt;script&gt;"));
  assert.ok(html.includes('data-source="1"'));
  assert.ok(!html.includes('data-source="7"'));
  assert.ok(html.includes("<ul><li>item"));
});
