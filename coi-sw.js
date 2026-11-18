// Cross-origin isolation service worker.
// GitHub Pages cannot send COOP/COEP headers, and without them the browser
// refuses SharedArrayBuffer, so WebAssembly inference runs on one thread.
// This worker adds the headers to this site's own responses. It is only
// registered on devices without WebGPU, where the CPU fallback model runs.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (new URL(req.url).origin !== self.location.origin) return; // cross-origin: untouched
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  event.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "credentialless");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    }),
  );
});
