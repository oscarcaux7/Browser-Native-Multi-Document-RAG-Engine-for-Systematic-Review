// IndexedDB persistence for the library. Usable from the page and from workers.
//   docs    : one record per PDF  {id, name, size, pages, addedAt}
//   files   : original PDF bytes  {id, blob}  (needed to show cited pages)
//   indexes : chunks + vectors per (embedding model, doc)
//             {key: `${model}|${docId}`, model, docId, chunks: [{page, text}], vectors: Float32Array}

const DB_NAME = "synapse";
const DB_VERSION = 1;
let dbPromise;

function open() {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore("docs", { keyPath: "id" });
      db.createObjectStore("files", { keyPath: "id" });
      const idx = db.createObjectStore("indexes", { keyPath: "key" });
      idx.createIndex("model", "model");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

const done = (req) =>
  new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

async function tx(stores, mode, fn) {
  const db = await open();
  const t = db.transaction(stores, mode);
  // Listen before running fn: a transaction can complete as soon as fn's last request does.
  const completed = new Promise((resolve, reject) => {
    t.oncomplete = resolve;
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
  const result = await fn(...[stores].flat().map((s) => t.objectStore(s)));
  await completed;
  return result;
}

export const store = {
  listDocs: () => tx("docs", "readonly", (s) => done(s.getAll())),
  getDoc: (id) => tx("docs", "readonly", (s) => done(s.get(id))),
  getFile: (id) => tx("files", "readonly", (s) => done(s.get(id))).then((r) => r?.blob),

  putDocWithFile: (doc, blob) =>
    tx(["docs", "files"], "readwrite", (docs, files) => {
      docs.put(doc);
      files.put({ id: doc.id, blob });
    }),

  putIndex: (model, docId, chunks, vectors) =>
    tx("indexes", "readwrite", (s) => s.put({ key: `${model}|${docId}`, model, docId, chunks, vectors })),

  indexesForModel: (model) => tx("indexes", "readonly", (s) => done(s.index("model").getAll(model))),

  removeDoc: (docId) =>
    tx(["docs", "files", "indexes"], "readwrite", async (docs, files, indexes) => {
      docs.delete(docId);
      files.delete(docId);
      const keys = await done(indexes.getAllKeys());
      for (const k of keys) if (k.endsWith(`|${docId}`)) indexes.delete(k);
    }),

  clearAll: () =>
    tx(["docs", "files", "indexes"], "readwrite", (docs, files, indexes) => {
      docs.clear();
      files.clear();
      indexes.clear();
    }),
};
