package info.skyblond.libllama.example

import info.skyblond.libllama.LibLLaMa
import info.skyblond.libllama.llama_context
import info.skyblond.libllama.llama_context_params
import info.skyblond.libllama.llama_model
import java.util.*
import kotlin.random.Random

@Suppress("ArrayInDataClass")
data class GPTParams(
    /**
     * RNG seed.
     * */
    val seed: Int = Random.nextInt(),
    /**
     * number of threads to use during computation
     * */
    val nThread: Int = (Runtime.getRuntime().availableProcessors() / 2).coerceAtLeast(1),

    /**
     * number of tokens to predict. -1 means infinity
     * */
    val nPredict: Int = -1,
    /**
     * size of the prompt context
     * */
    val contextSize: Int = 512,
    /**
     * batch size for prompt processing
     * */
    val batchSize: Int = 512,
    /**
     * grouped-query attention factor.
     * default 1, for LLaMa 2 70B, use 8
     * */
    val gqa: Int = 1,
    /**
     * number of tokens to keep from initial prompt
     * */
    val nKeep: Int = 0,
    /**
     * max number of chunks to process (-1 = unlimited)
     * */
    val nChunks: Int = -1,
    /**
     * number of layers to store in VRAM
     * */
    val nGpuLayers: Int = 0,
    /**
     * the GPU that is used for scratch and small tensors
     * */
    val mainGpu: Int = 0,
    /**
     * how split tensors should be distributed across GPUs
     * */
    val tensorSplit: FloatArray = floatArrayOf(0.0f),
    /**
     * if greater than 0, output the probabilities of top n_probs tokens.
     * */
    val nProbs: Int = 0,
    /**
     * N rms norm eps.
     * default: 5.0e-06; use 1e-5 for LLaMAv2.
     * */
    val rmsNormEps: Float = 5e-6f,
    /**
     * RoPE base frequency
     * */
    val ropeFreqBase: Float = 10000.0f,
    /**
     * RoPE frequency scaling factor
     * */
    val ropeFreqScale: Float = 1.0f,

    // sampling parameters
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

    // Classifier-Free Guidance
    /**
     * string to help guidance. negative prompt to use for guidance.
     * */
    val cfgNegativePrompt: String = "",
    /**
     * strength of guidance. 1.0 means disabled
     * */
    val cfgScale: Float = 1.0f,

    /**
     * The file path of the model.
     * */
    val modelPath: String,
    /**
     * model alias
     * */
    val modelAlias: String = "unknown",
    /**
     * The initial prompt
     * */
    val prompt: String = "",
    /**
     * The path of prompt cache (eval state)
     * */
    val pathPromptCache: String = "",
    /**
     * String to prefix user input with
     * */
    val inputPrefix: String = "",
    /**
     * String to suffix user inputs with
     * */
    val inputSuffix: String = "",
//    /**
//     * optional BNF-like grammar to constrain sampling
//     * */
//    val grammar: String = "",
    /**
     * Reverse prompt. halt generation at PROMPT, return control in interactive mode
     * */
    val antiprompt: MutableList<String> = LinkedList(),

    /**
     * apply LoRA adapter
     * */
    val loraAdapter: String = "",
    /**
     * optional model to use as a base for the layers modified by the LoRA adapter
     * */
    val loraBase: String = "",

    /**
     * compute HellaSwag score over random tasks from datafile supplied in prompt
     * */
    val hellaswag: Boolean = false,
    /**
     * number of tasks to use when computing the HellaSwag score
     * */
    val hellaswag_tasks: Int = 400,

    /**
     * if true, reduce VRAM usage at the cost of performance
     * */
    val lowVram: Boolean = false,
    /**
     * use f16 instead of f32 for memory kv
     * */
    val memoryUseF16: Boolean = true,
    /**
     * do not randomize prompt if none provided
     * */
    val randomPrompt: Boolean = false,
    /**
     * save user input and generations to prompt cache
     * */
    val promptCacheAll: Boolean = false,
    /**
     * open the prompt cache read-only and do not update it
     * */
    val promptCacheReadOnly: Boolean = false,

    /**
     * prefix BOS to user inputs, preceding input_prefix
     * */
    val inputPrefixBOS: Boolean = false,
    /**
     * instruction mode (used for Alpaca models)
     * */
    val instruct: Boolean = false,
    /**
     * consider newlines as a repeatable token
     * */
    val penalizeNewLine: Boolean = true,
    /**
     * use mmap for faster loads. Disabled this will result in a slow loading
     * but may reduce pageouts if not using memory lock.
     * */
    val useMemoryMap: Boolean = true,
    /**
     * force system to keep model in RAM rather than swapping or compressing
     * */
    val useMemoryLock: Boolean = false,
    /**
     * attempt optimizations that help on some NUMA systems
     * if run without this previously, it is recommended to drop the system page cache before using this
     * see https://github.com/ggerganov/llama.cpp/issues/1437
     * */
    val numa: Boolean = false,
)

fun Boolean.toJNAByte(): Byte = if (this) 1 else 0

fun LibLLaMa.llama_context_params_from_gpt_params(
    params: GPTParams,
    // test perplexity when true
    perplexity: Boolean = false,
    // get embedding when true
    embedding: Boolean = false
): llama_context_params.ByValue {
    val llamaParams = llama_context_default_params()

    llamaParams.n_ctx = params.contextSize
    llamaParams.n_batch = params.batchSize
    llamaParams.n_gqa = params.gqa
    llamaParams.rms_norm_eps = params.rmsNormEps
    llamaParams.n_gpu_layers = params.nGpuLayers
    llamaParams.main_gpu = params.mainGpu
    llamaParams.tensor_split = params.tensorSplit
    llamaParams.low_vram = params.lowVram.toJNAByte()
    llamaParams.seed = params.seed
    llamaParams.f16_kv = params.memoryUseF16.toJNAByte()
    llamaParams.use_mmap = params.useMemoryMap.toJNAByte()
    llamaParams.use_mlock = params.useMemoryLock.toJNAByte()
    llamaParams.logits_all = perplexity.toJNAByte()
    llamaParams.embedding = embedding.toJNAByte()
    llamaParams.rope_freq_base = params.ropeFreqBase
    llamaParams.rope_freq_scale = params.ropeFreqScale

    return llamaParams
}


fun LibLLaMa.llama_init_from_gpt_params(
    params: GPTParams,
    perplexity: Boolean = false, embedding: Boolean = false
): Pair<llama_model, llama_context> {
    val lparams = llama_context_params_from_gpt_params(params, perplexity, embedding)
    val model = llama_load_model_from_file(params.modelPath, lparams)
        ?: error("failed to load model: ${params.modelPath}")

    val ctx = llama_new_context_with_model(model, lparams)
        ?: run {
            llama_free_model(model)
            error("failed to create context with model: ${params.modelPath}")
        }

    if (params.loraAdapter.isNotBlank()) {
        val err = llama_model_apply_lora_from_file(
            model, params.loraAdapter, params.loraBase.ifBlank { null }, params.nThread
        )
        if (err != 0) {
            llama_free(ctx)
            llama_free_model(model)
            error("failed to apply lora adapter")
        }
    }

    return model to ctx
}
