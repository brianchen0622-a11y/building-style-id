// Caches the app shell only; model downloads (CDN/HuggingFace) stay
// network-only since transformers.js manages its own model cache.
const CACHE = "building-style-shell-v3";
const SHELL = [
  "./",
  "./index.html",
  "./style.css?v=7",
  "./app.js?v=12",
  "./manifest.json",
  "./data/styles.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== location.origin) return; // let CDN/model requests pass through untouched

  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
