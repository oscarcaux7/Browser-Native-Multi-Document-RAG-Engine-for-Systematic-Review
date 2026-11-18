// Central configuration: pinned library versions and model catalogues.
// Versions and model ids were checked against the published npm packages and
// the Hugging Face Hub (November 2026). Bump them together, then re-test.

export const LIBS = {
  webllm: "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.88/+esm",
  transformers: "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2",
  pdfjs: "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.min.mjs",
  pdfjsWorker: "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.worker.min.mjs",
};

// Chat models served by WebLLM (WebGPU). Ids come from WebLLM 0.2.88's
// prebuiltAppConfig; download sizes from the mlc-ai repos on Hugging Face,
// VRAM figures from WebLLM's own model records.
export const GPU_MODELS = [
  {
    id: "Qwen3.5-0.8B-q4f16_1-MLC",
    f32: "Qwen3.5-0.8B-q4f32_1-MLC",
    label: "Qwen3.5 0.8B",
    note: "Fastest. Phones and integrated GPUs.",
    download: "≈0.45 GB",
    vram: "1.6 GB",
    thinking: true,
  },
  {
    id: "Qwen3.5-2B-q4f16_1-MLC",
    f32: "Qwen3.5-2B-q4f32_1-MLC",
    label: "Qwen3.5 2B",
    note: "Balanced default for laptops.",
    download: "≈1.1 GB",
    vram: "2.2 GB",
    thinking: true,
  },
  {
    id: "Qwen3.5-4B-q4f16_1-MLC",
    f32: "Qwen3.5-4B-q4f32_1-MLC",
    label: "Qwen3.5 4B",
    note: "Best answers, needs a discrete GPU.",
    download: "≈2.4 GB",
    vram: "3.9 GB",
    thinking: true,
  },
  {
    id: "gemma-4-E2B-it-q4f16_1-MLC",
    f32: "gemma-4-E2B-it-q4f32_1-MLC",
    label: "Gemma 4 E2B",
    note: "Strong multilingual model.",
    download: "≈2.8 GB",
    vram: "3.4 GB",
    thinking: false,
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    f32: "Llama-3.2-3B-Instruct-q4f32_1-MLC",
    label: "Llama 3.2 3B",
    note: "Well-known baseline.",
    download: "≈1.8 GB",
    vram: "2.3 GB",
    thinking: false,
  },
];

export const DEFAULT_GPU_MODEL = "Qwen3.5-2B-q4f16_1-MLC";
export const LIGHT_GPU_MODEL = "Qwen3.5-0.8B-q4f16_1-MLC";

// Used when WebGPU is unavailable: transformers.js on WebAssembly (CPU).
export const CPU_MODEL = {
  id: "onnx-community/Qwen3-0.6B-ONNX",
  dtype: "q8",
  label: "Qwen3 0.6B (CPU)",
  note: "Runs on the CPU via WebAssembly. Slower, but works without WebGPU.",
  download: "≈0.6 GB",
  thinking: true,
};

// Embedding models run with transformers.js (WebGPU when available, else WASM).
// Add an entry here to offer another model; switching re-indexes the library.
// queryPrefix / docPrefix follow each model card's recommended prompts.
export const EMBEDDING_MODELS = [
  {
    id: "Xenova/bge-small-en-v1.5",
    label: "BGE small v1.5 (English)",
    note: "384-d, fast. Best for English papers.",
    download: "≈35–65 MB",
    dims: 384,
    dtype: { webgpu: "fp16", webgpuNoF16: "fp32", wasm: "q8" },
    pooling: "cls", // BGE uses the [CLS] token embedding
    maxTokens: 512,
    // Best passage below this cosine similarity = probably off-topic (measured on the sample papers:
    // relevant questions scored 0.63–0.85, an unrelated one 0.48).
    lowRelevance: 0.52,
    queryPrefix: "Represent this sentence for searching relevant passages: ",
    docPrefix: () => "",
  },
];

export const DEFAULT_EMBEDDING_MODEL = "Xenova/bge-small-en-v1.5";

export const CHUNKING = { targetChars: 900, overlapChars: 180, minChars: 40 };

export const DEFAULT_SETTINGS = {
  theme: "system", // "system" | "light" | "dark"
  gpuModel: null, // null = auto-pick on first run
  embeddingModel: DEFAULT_EMBEDDING_MODEL,
  temperature: 0.2,
  topK: 6,
  maxTokens: 768,
};

export function findGpuModel(id) {
  return GPU_MODELS.find((m) => m.id === id || m.f32 === id);
}

export function findEmbeddingModel(id) {
  return EMBEDDING_MODELS.find((m) => m.id === id) || EMBEDDING_MODELS[0];
}
