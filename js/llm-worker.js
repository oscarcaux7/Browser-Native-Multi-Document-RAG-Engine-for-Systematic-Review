// Chat model worker. Two back-ends behind one message protocol:
//   ?backend=webgpu  -> WebLLM (MLC, WebGPU), fast, quantised 0.8B–4B models
//   ?backend=wasm    -> transformers.js on WebAssembly (CPU) fallback
// Protocol (in):  {type:"load", model} {type:"generate", requestId, messages, temperature, maxTokens, disableThinking}
//                 {type:"interrupt"} {type:"cache-status", models}
// Protocol (out): {type:"load-progress", progress, text} {type:"loaded", model, seconds}
//                 {type:"token", requestId, text} {type:"done", requestId, stats} {type:"error", ...}
import { LIBS, CPU_MODEL } from "./config.js";

const backend = new URL(self.location.href).searchParams.get("backend") || "webgpu";
const post = (m) => self.postMessage(m);

// --------------------------------------------------------------- WebGPU (WebLLM)
function webgpuBackend(webllm) {
  let engine = null;
  return {
    async load(model) {
      const progress = (r) => post({ type: "load-progress", progress: r.progress, text: r.text });
      if (engine) await engine.reload(model);
      else engine = await webllm.CreateMLCEngine(model, { initProgressCallback: progress });
    },
    async generate({ requestId, messages, temperature, maxTokens, disableThinking }) {
      const stream = await engine.chat.completions.create({
        messages,
        temperature,
        max_tokens: maxTokens,
        stream: true,
        stream_options: { include_usage: true },
        ...(disableThinking ? { extra_body: { enable_thinking: false } } : {}), // Qwen: answer directly
      });
      let usage = null;
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) post({ type: "token", requestId, text: delta });
        if (chunk.usage) usage = chunk.usage;
      }
      return usage
        ? {
            promptTokens: usage.prompt_tokens,
            completionTokens: usage.completion_tokens,
            decodeTps: usage.extra?.decode_tokens_per_s,
            prefillTps: usage.extra?.prefill_tokens_per_s,
            ttft: usage.extra?.time_to_first_token_s,
          }
        : null;
    },
    interrupt: () => engine?.interruptGenerate(),
    async cacheStatus(models) {
      const out = {};
      for (const m of models) out[m] = await webllm.hasModelInCache(m).catch(() => false);
      return out;
    },
  };
}

// --------------------------------------------------------- CPU (transformers.js)
function wasmBackend(tf) {
  // With cross-origin isolation (see coi-sw.js) ONNX Runtime can use several threads.
  if (self.crossOriginIsolated) tf.env.backends.onnx.wasm.numThreads = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 4) - 2));
  let tokenizer = null;
  let model = null;
  const stopper = new tf.InterruptableStoppingCriteria();
  return {
    async load() {
      const progress_callback = (p) => {
        if (p.status === "progress_total" && p.total)
          post({ type: "load-progress", progress: p.loaded / p.total, text: `Downloading ${CPU_MODEL.label}` });
      };
      tokenizer ??= await tf.AutoTokenizer.from_pretrained(CPU_MODEL.id, { progress_callback });
      model ??= await tf.AutoModelForCausalLM.from_pretrained(CPU_MODEL.id, {
        dtype: CPU_MODEL.dtype,
        device: "wasm",
        progress_callback,
      });
    },
    async generate({ requestId, messages, temperature, maxTokens }) {
      const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
        enable_thinking: false, // Qwen3 template switch: answer directly
      });
      let first = 0;
      let count = 0;
      const t0 = performance.now();
      const streamer = new tf.TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => post({ type: "token", requestId, text }),
        token_callback_function: () => {
          count++;
          if (!first) first = performance.now();
        },
      });
      stopper.reset();
      await model.generate({
        ...inputs,
        max_new_tokens: maxTokens,
        do_sample: temperature > 0,
        temperature: Math.max(temperature, 0.01),
        top_p: 0.9,
        repetition_penalty: 1.1, // tiny CPU model: discourage loops
        streamer,
        stopping_criteria: stopper,
      });
      const end = performance.now();
      return {
        promptTokens: inputs.input_ids.dims.at(-1),
        completionTokens: count,
        ttft: first ? (first - t0) / 1000 : null,
        decodeTps: count > 1 && first ? (count - 1) / ((end - first) / 1000) : null,
      };
    },
    interrupt: () => stopper.interrupt(),
    async cacheStatus() {
      return {};
    },
  };
}

const impl =
  backend === "wasm"
    ? wasmBackend(await import(LIBS.transformers))
    : webgpuBackend(await import(LIBS.webllm));

self.onmessage = async ({ data: msg }) => {
  try {
    switch (msg.type) {
      case "load": {
        const t0 = performance.now();
        await impl.load(msg.model);
        post({ type: "loaded", model: msg.model, seconds: (performance.now() - t0) / 1000 });
        break;
      }
      case "generate": {
        const stats = await impl.generate(msg);
        post({ type: "done", requestId: msg.requestId, stats });
        break;
      }
      case "interrupt":
        impl.interrupt();
        break;
      case "cache-status":
        post({ type: "cache-status", cached: await impl.cacheStatus(msg.models) });
        break;
    }
  } catch (err) {
    console.error(err);
    const text = err?.message || String(err);
    // A lost GPU device (driver reset, out of memory) leaves the engine unusable:
    // tell the page so it can restart this worker and reload from cache.
    const fatal = /device (was )?lost|mapAsync|already been disposed|ModelNotLoaded|Model not loaded/i.test(text);
    post({ type: "error", requestId: msg.requestId, during: msg.type, error: text, fatal });
  }
};

post({ type: "booted", backend });
