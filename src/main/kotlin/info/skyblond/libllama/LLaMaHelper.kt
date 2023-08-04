package info.skyblond.libllama

import com.sun.jna.Pointer
import kotlin.random.Random


fun Boolean.toJNAByte(): Byte = if (this) 1 else 0

/**
 * Init the llama backend.
 * @param numa set to true if there is a numa system.
 * @see [LibLLaMa.llama_backend_init]
 * @see [LibLLaMa.llama_backend_free]
 * */
fun LibLLaMa.initLLaMaBackend(numa: Boolean = false) = llama_backend_init(if (numa) 1 else 0)

typealias ContextParams = llama_context_params.ByValue

/**
 * Load default parameter and override it using supplied value.
 * */
fun LibLLaMa.getContextParams(
    /**
     * RNG seed.
     * */
    seed: Int = Random.nextInt(),
    /**
     * context window size
     * */
    contextSize: Int = 512,
    /**
     * batch size for prompt processing
     * */
    batchSize: Int = 512,
    /**
     * grouped-query attention factor.
     * default 1, for LLaMa 2 70B, use 8
     * */
    gqa: Int = 1,
    /**
     * N rms norm eps.
     * default: 5.0e-06; use 1e-5 for LLaMAv2.
     * */
    rmsNormEps: Float = 5e-6f,
    /**
     * number of layers to store in VRAM
     * */
    nGpuLayers: Int = 0,
    /**
     * the GPU that is used for scratch and small tensors
     * */
    mainGpu: Int = 0,
    /**
     * how split tensors should be distributed across GPUs
     * */
    tensorSplit: FloatArray = floatArrayOf(0.0f),
    /**
     * RoPE base frequency
     * */
    ropeFreqBase: Float = 10000.0f,
    /**
     * RoPE frequency scaling factor
     * */
    ropeFreqScale: Float = 1.0f,
    /**
     * called with a progress value between 0 and 1, pass NULL to disable
     * */
    progressCallback: llama_progress_callback = llama_progress_callback { _, _ -> },
    /**
     * context pointer passed to the progress callback
     * */
    progressCallbackUserData: Pointer? = Pointer.NULL,
    /**
     * if true, reduce VRAM usage at the cost of performance
     * */
    lowVram: Boolean = false,
    /**
     * if true, use experimental mul_mat_q kernels
     * */
    useMulMatQ: Boolean = false,
    /**
     * use f16 instead of f32 for memory kv
     * */
    memoryUseF16: Boolean = true,
    /**
     * use mmap for faster loads. Disabled this will result in a slow loading
     * but may reduce pageouts if not using memory lock.
     * */
    useMemoryMap: Boolean = true,
    /**
     * force system to keep model in RAM rather than swapping or compressing
     * */
    useMemoryLock: Boolean = false,
    /**
     * Set to true when testing perplexity.
     * This will make the llama_eval() call computes all logits instead of only the last one
     * */
    logitsAll: Boolean = false,
    /**
     * only load the vocabulary, no weights
     * */
    vocabOnly: Boolean = false,
    /**
     * Set to true when getting embedding
     * */
    embedding: Boolean = false,
): ContextParams = llama_context_default_params().apply {
    this.seed = seed
    this.n_ctx = contextSize.coerceAtLeast(8)
    this.n_batch = batchSize
    this.n_gqa = gqa
    this.rms_norm_eps = rmsNormEps
    this.n_gpu_layers = nGpuLayers
    this.main_gpu = mainGpu
    this.tensor_split = tensorSplit
    this.rope_freq_base = ropeFreqBase
    this.rope_freq_scale = ropeFreqScale
    this.progress_callback = progressCallback
    this.progress_callback_user_data = progressCallbackUserData
    this.low_vram = lowVram.toJNAByte()
    this.mul_mat_q = useMulMatQ.toJNAByte()
    this.f16_kv = memoryUseF16.toJNAByte()
    this.logits_all = logitsAll.toJNAByte()
    this.vocab_only = vocabOnly.toJNAByte()
    this.use_mmap = useMemoryMap.toJNAByte()
    this.use_mlock = useMemoryLock.toJNAByte()
    this.embedding = embedding.toJNAByte()
    this.write() // sync jvm side change to native c side
}

/**
 * Parameters that instruct how to load a model.
 * */
data class ModelParams(
    /**
     * The file path of the model.
     * */
    val modelPath: String,
    /**
     * model alias
     * */
    val modelAlias: String = "unknown",
    /**
     * apply LoRA adapter
     * */
    val loraAdapter: String = "",
    /**
     * optional model to use as a base for the layers modified by the LoRA adapter
     * */
    val loraBase: String = "",
)


/**
 * Load [llama_model] amd [llama_context] with parameters.
 * Support lora.
 * */
fun LibLLaMa.loadModelAndContextWithParams(
    llamaParams: ContextParams, modelParams: ModelParams,
    nThread: Int = getProcessorCount()
): Pair<llama_model, llama_context> {
    val model = llama_load_model_from_file(modelParams.modelPath, llamaParams)
        ?: error("failed to load model: ${modelParams.modelPath}")

    val ctx = llama_new_context_with_model(model, llamaParams)
        ?: run {
            llama_free_model(model)
            error("failed to create context with model: ${modelParams.modelPath}")
        }

    if (modelParams.loraAdapter.isNotBlank()) {
        val err = llama_model_apply_lora_from_file(
            model, modelParams.loraAdapter, modelParams.loraBase.ifBlank { null }, nThread
        )
        if (err != 0) {
            llama_free(ctx)
            llama_free_model(model)
            error("failed to apply lora adapter")
        }
    }

    return model to ctx
}

/**
 * Turn [text] into token array. Use [ctx] to decide vocabulary.
 * use [tokenizeWithModel] if you want to use model to decide vocabulary.
 * */
fun LibLLaMa.tokenize(ctx: llama_context, text: String, addBOS: Boolean = false): IntArray {
    // the biggest size possible
    val res = IntArray(text.toByteArray().size + if (addBOS) 1 else 0)
    val n = this.llama_tokenize(ctx, text, res, res.size, addBOS.toJNAByte())
    check(n >= 0) { "Failed to tokenize: code $n, text: \"$text\", bos: $addBOS" }

    return res.copyOf(n)
}

fun LibLLaMa.tokenizeWithModel(model: llama_model, text: String, addBOS: Boolean = false): IntArray {
    // the biggest size possible
    val res = IntArray(text.toByteArray().size + if (addBOS) 1 else 0)
    val n = this.llama_tokenize_with_model(model, text, res, res.size, if (addBOS) 1 else 0)
    check(n >= 0) { "Failed to tokenize with model: code $n, text: \"$text\", bos: $addBOS" }

    return res.copyOf(n)
}

/**
 * Evaluate the input [tokens] using [ctx].
 * @param nPast decide the context size when generating the next token, must not bigger than ctxSize.
 * */
fun LibLLaMa.evalTokens(
    ctx: llama_context, tokens: IntArray, nPast: Int, nThread: Int
): Unit = check(llama_eval(ctx, tokens, tokens.size, nPast, nThread) == 0) { "Failed to eval" }

/**
 * Evaluate the input [tokens] using [ctx].
 * @param nPast decide the context size when generating the next token, must not bigger than ctxSize.
 * */
fun LibLLaMa.evalTokens(
    ctx: llama_context, tokens: List<Int>, nPast: Int, nThread: Int
) = evalTokens(ctx, tokens.toIntArray(), nPast, nThread)

/**
 * Get the processor number. Considering most systems are using hyper threading,
 * thus the result will divide by 2.
 * */
fun getProcessorCount() = Runtime.getRuntime().availableProcessors().let {
    // if we have less than 8 core, then use it - 1 thread, leaving the last one to system
    // otherwise we take hyper threading into account and use half of them
    if (it <= 8) (it - 1).coerceAtLeast(1) else it / 2
}