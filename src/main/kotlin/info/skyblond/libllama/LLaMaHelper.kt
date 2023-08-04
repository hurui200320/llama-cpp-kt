package info.skyblond.libllama

import com.sun.jna.Pointer
import com.sun.jna.ptr.FloatByReference
import java.util.*
import kotlin.math.min
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
 * use [tokenizeWithModel] if you want to use [llama_model] to decide vocabulary.
 * */
fun LibLLaMa.tokenize(ctx: llama_context, text: String, addBOS: Boolean = false): IntArray {
    // the biggest size possible
    val res = IntArray(text.toByteArray().size + if (addBOS) 1 else 0)
    val n = this.llama_tokenize(ctx, text, res, res.size, addBOS.toJNAByte())
    check(n >= 0) { "Failed to tokenize: code $n, text: \"$text\", bos: $addBOS" }

    return res.copyOf(n)
}

/**
 * Turn [text] into token array. Use [model] to decide vocabulary.
 * use [tokenize] if you want to use [llama_context] to decide vocabulary.
 * */
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
 * Parameters for sampling.
 * @see sampleToken
 * */
data class SamplingParams(
    /**
     * logit bias for specific tokens.
     * For example: 15043 to 1 will increase likelihood of token ' Hello';
     * use 15043 to -1 will decrease likelihood of token ' Hello'
     * */
    val logitBias: MutableMap<Int, Float> = HashMap(),
    /**
     * top-k sampling. Smaller or equal than 0 to use vocab size
     * */
    val topK: Int = 40,
    /**
     * top-p sampling. 1.0 = disabled
     * */
    val topP: Float = 0.95f,
    /**
     * tail free sampling, parameter z. 1.0 = disabled
     * */
    val tfsZ: Float = 1.00f,
    /**
     * locally typical sampling, parameter p. 1.0 = disabled
     * */
    val typicalP: Float = 1.00f,
    /**
     * temperature. 1.0 = disabled
     * */
    val temp: Float = 0.80f,
    /**
     * penalize repeat sequence of tokens. 1.0 = disabled.
     * Bigger than 1.0 = discourage repeat.
     * Smaller than 1.0 = encourage repeat.
     * */
    val repeatPenalty: Float = 1.10f,
    /**
     * last n tokens to consider for penalize.
     * 0 means disabled; -1 means ctx size.
     * */
    val repeatLastN: Int = 64,
    /**
     * consider newlines as a repeatable token
     * */
    val penalizeNewLine: Boolean = true,
    /**
     * repeat alpha frequency penalty. 0.0 = disabled
     * */
    val frequencyPenalty: Float = 0.00f,
    /**
     * repeat alpha presence penalty. 0.0 = disabled
     * */
    val presencePenalty: Float = 0.00f,
    /**
     * use Mirostat sampling.
     * Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
     * 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0
     * */
    val mirostat: Int = 0,
    /**
     * Mirostat target entropy, parameter tau
     * */
    val mirostatTau: Float = 5.00f,
    /**
     * Mirostat learning rate, parameter eta
     * */
    val mirostatEta: Float = 0.10f,
)

/**
 * Ring buffer for storing last n tokens.
 * Head -> Tail
 * Old -> New
 *
 * NOT THREAD SAFE!
 * */
class RingTokenBuffer(
    override val size: Int,
    initValue: Int = 0
) : MutableList<Int> {
    private val buffer = LinkedList<Int>()

    init {
        for (i in 0 until size)
            buffer.add(initValue)
    }

    override fun contains(element: Int): Boolean = buffer.contains(element)

    override fun containsAll(elements: Collection<Int>): Boolean = buffer.containsAll(elements)

    private fun ensureSize() {
        while (buffer.size > size) buffer.removeFirst()
    }

    override fun add(element: Int): Boolean =
        buffer.add(element).also { ensureSize() }

    override fun add(index: Int, element: Int) =
        buffer.add(index, element).also { ensureSize() }

    override fun addAll(index: Int, elements: Collection<Int>): Boolean =
        buffer.addAll(index, elements).also { ensureSize() }

    override fun addAll(elements: Collection<Int>): Boolean =
        buffer.addAll(elements).also { ensureSize() }

    override fun clear() = buffer.clear()

    override fun get(index: Int): Int = buffer.get(index)

    override fun isEmpty(): Boolean = buffer.isEmpty()

    override fun iterator(): MutableIterator<Int> = buffer.iterator()

    override fun listIterator(): MutableListIterator<Int> = buffer.listIterator()

    override fun listIterator(index: Int): MutableListIterator<Int> = buffer.listIterator(index)

    override fun removeAt(index: Int): Int = buffer.removeAt(index)

    override fun set(index: Int, element: Int): Int = buffer.set(index, element)

    override fun retainAll(elements: Collection<Int>): Boolean = buffer.retainAll(elements)

    override fun removeAll(elements: Collection<Int>): Boolean = buffer.removeAll(elements)

    override fun remove(element: Int): Boolean = buffer.remove(element)

    override fun subList(fromIndex: Int, toIndex: Int): MutableList<Int> = buffer.subList(fromIndex, toIndex)

    override fun lastIndexOf(element: Int): Int = buffer.lastIndexOf(element)

    override fun indexOf(element: Int): Int = buffer.indexOf(element)
}

fun LibLLaMa.sampleToken(
    ctx: llama_context,
    samplingParams: SamplingParams,
    lastNTokens: RingTokenBuffer,
    guidanceCtx: llama_context = null,
    guidanceScale: Float = 1.0f
): Int {
    val ctxSize = llama_n_ctx(ctx)

    // select best prediction
    val nVocab = llama_n_vocab(ctx)
    val logits = llama_get_logits(ctx).getFloatArray(0, nVocab)

    // Apply logit_bias map
    samplingParams.logitBias.forEach { (t, v) ->
        logits[t] += v
    }

    val candidates = llama_token_data_array.ByReference().apply {
        // here we need an array of llama_token_data,
        // which must act like a single blob of memory.
        // thus we use `toArray(Int)` to get that
        val candidates = llama_token_data().toArray(nVocab)
        candidates.forEachIndexed { index, structure ->
            val obj = structure as llama_token_data
            obj.id = index
            obj.logit = logits[index]
            obj.p = 0.0f
            // don't forget to write or the changes won't be seen in native c side
            obj.write()
        }
        // since the candidates is a continues blob of memory
        // we can use the first element's pointer as the array pointer
        data = candidates[0].pointer
        size = candidates.size
        sorted = 0
    }

    if (guidanceCtx != null) {
        llama_sample_classifier_free_guidance(ctx, candidates, guidanceCtx, guidanceScale)
    }

    // Apply penalties
    val newlineLogit = logits[llama_token_nl()]
    val lastNRepeat = min(min(lastNTokens.size, samplingParams.repeatLastN), ctxSize)
    llama_sample_repetition_penalty(
        ctx, candidates,
        lastNTokens.drop(lastNTokens.size - lastNRepeat).toIntArray(),
        lastNRepeat, samplingParams.repeatPenalty
    )
    llama_sample_frequency_and_presence_penalties(
        ctx, candidates,
        lastNTokens.drop(lastNTokens.size - lastNRepeat).toIntArray(),
        lastNRepeat,
        samplingParams.frequencyPenalty, samplingParams.presencePenalty
    )
    if (!samplingParams.penalizeNewLine) {
        logits[llama_token_nl()] = newlineLogit
    }

    // TODO: grammar?
    // if (grammar != NULL) {
    //     llama_sample_grammar(ctx, candidates, grammar);
    // }

    val tokenId = if (samplingParams.temp <= 0) {
        // Greedy sampling
        llama_sample_token_greedy(ctx, candidates)
    } else {
        if (samplingParams.mirostat == 1) {
            val mirostatMu = FloatByReference(2.0f * samplingParams.mirostatTau)
            val mirostatM = 100
            llama_sample_temperature(ctx, candidates, samplingParams.temp)
            llama_sample_token_mirostat(
                ctx, candidates, samplingParams.mirostatTau, samplingParams.mirostatEta,
                mirostatM, mirostatMu
            )
        } else if (samplingParams.mirostat == 2) {
            val mirostatMu = FloatByReference(2.0f * samplingParams.mirostatTau)
            llama_sample_temperature(ctx, candidates, samplingParams.temp)
            llama_sample_token_mirostat_v2(
                ctx, candidates, samplingParams.mirostatTau, samplingParams.mirostatEta,
                mirostatMu
            )
        } else {
            // Temperature sampling
            llama_sample_top_k(ctx, candidates, samplingParams.topK, 1)
            llama_sample_tail_free(ctx, candidates, samplingParams.tfsZ, 1)
            llama_sample_typical(ctx, candidates, samplingParams.typicalP, 1)
            llama_sample_top_p(ctx, candidates, samplingParams.topP, 1)
            llama_sample_temperature(ctx, candidates, samplingParams.temp)
            llama_sample_token(ctx, candidates)
        }
    }

    // TODO: grammar?
    // if (grammar != NULL) {
    //     llama_grammar_accept_token(ctx, grammar, id);
    // }

    // remove the head, aka the oldest one
    lastNTokens.removeFirst()
    // add this
    lastNTokens.add(tokenId)

    return tokenId
}

/**
 * Get the processor number. Considering most systems are using hyper threading,
 * thus the result will divide by 2.
 * */
fun getProcessorCount() = Runtime.getRuntime().availableProcessors().let {
    // if we have less than 8 core, then use it - 1 thread, leaving the last one to system
    // otherwise we take hyper threading into account and use half of them
    if (it <= 8) (it - 1).coerceAtLeast(1) else it / 2
}