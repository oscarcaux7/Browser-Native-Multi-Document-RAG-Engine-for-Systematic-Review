// Minimal, safe Markdown renderer for model answers. Everything is HTML-escaped
// first; only a small whitelist of formatting is turned back into tags.
import { CITE_RE, expandCitation } from "./prompt.js";

export function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function inline(text, sourceCount) {
  let s = escapeHtml(text);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  s = s.replace(CITE_RE, (match, inner) => {
    const nums = expandCitation(inner).filter((n) => n >= 1 && n <= sourceCount);
    if (!nums.length) return match;
    return nums
      .map((n) => `<button type="button" class="cite" data-source="${n}" aria-label="Open source ${n}">${n}</button>`)
      .join("");
  });
  return s;
}

/**
 * @param {string} md
 * @param {number} sourceCount number of sources that citations may refer to
 */
export function renderMarkdown(md, sourceCount = 0) {
  const lines = md.replace(/\r/g, "").split("\n");
  const html = [];
  let list = null; // "ul" | "ol"
  let para = [];
  let code = null;

  const flushPara = () => {
    if (para.length) html.push(`<p>${inline(para.join(" "), sourceCount)}</p>`);
    para = [];
  };
  const closeList = () => {
    if (list) html.push(`</${list}>`);
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (code !== null) {
      if (/^```/.test(line)) {
        html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = null;
      } else code.push(raw);
      continue;
    }
    if (/^```/.test(line)) {
      flushPara();
      closeList();
      code = [];
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (heading) {
      flushPara();
      closeList();
      html.push(`<h4>${inline(heading[2], sourceCount)}</h4>`);
    } else if (bullet || numbered) {
      flushPara();
      const kind = bullet ? "ul" : "ol";
      if (list !== kind) {
        closeList();
        html.push(`<${kind}>`);
        list = kind;
      }
      html.push(`<li>${inline((bullet || numbered)[1], sourceCount)}</li>`);
    } else if (!line.trim()) {
      flushPara();
      closeList();
    } else {
      if (list) closeList();
      para.push(line.trim());
    }
  }
  if (code !== null) html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  flushPara();
  closeList();
  return html.join("");
}
