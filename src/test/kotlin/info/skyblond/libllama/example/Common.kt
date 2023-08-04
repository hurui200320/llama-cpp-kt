package info.skyblond.libllama.example

import java.util.*


/**
 * Parameters that instruct how to do inference.
 * */
data class InferenceParams(
    /**
     * number of tokens to predict (aka when to stop generating). -1 means infinity
     * */
    val nPredict: Int = -1,
    /**
     * number of tokens to keep when run out of context.
     * */
    val nKeep: Int = 0,
)

data class GPTParams(

    /**
     * max number of chunks to process (-1 = unlimited)
     * */
    val nChunks: Int = -1,


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
     * compute HellaSwag score over random tasks from datafile supplied in prompt
     * */
    val hellaswag: Boolean = false,
    /**
     * number of tasks to use when computing the HellaSwag score
     * */
    val hellaswag_tasks: Int = 400,


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
     * attempt optimizations that help on some NUMA systems
     * if run without this previously, it is recommended to drop the system page cache before using this
     * see https://github.com/ggerganov/llama.cpp/issues/1437
     * */
    val numa: Boolean = false,
)




