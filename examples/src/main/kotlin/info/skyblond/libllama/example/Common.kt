package info.skyblond.libllama.example


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
    /**
     * String to prefix user input with
     * */
    val inputPrefix: String = "",
    /**
     * String to suffix user inputs with
     * */
    val inputSuffix: String = "",
    /**
     * optional BNF-like grammar to constrain sampling.
     * TODO: figure out the grammar parser and how it works in llama
     * */
    val grammar: String = "",
    /**
     * Reverse prompt. halt generation at PROMPT, return control in interactive mode
     * */
    val reversePrompts: MutableList<String> = mutableListOf(),
    /**
     * prefix BOS to user inputs, preceding input_prefix
     * */
    val inputPrefixBOS: Boolean = false,
    /**
     * Set to true if llama has to wait for user input during the generation
     * */
    val interactive: Boolean = false,
    /**
     * Set to true if waiting for user input right after processed the prompt
     * */
    val interactiveFirst: Boolean = false,
    /**
     * instruction mode (used for Alpaca models)
     * */
    val alpacaInstruct: Boolean = false,
) {
    init {
        if (alpacaInstruct)
            check(interactiveFirst) { "Interactive First must be enabled when alpaca instruct mode is enabled" }
        if (interactiveFirst)
            check(interactive) { "Interactive mode must be enabled when interactive first is enabled" }
    }
}

/**
 * Parameters for negative guidance prompt. Aka words you want to avoid.
 * */
data class GuidanceParams(
    /**
     * string to help guidance. negative prompt to use for guidance.
     * */
    val negativePrompt: String,
    /**
     * strength of guidance. 1.0 means disabled
     * */
    val scale: Float = 1.0f,
)


data class PersistenceParams(
    /**
     * The path of prompt cache (eval state)
     * */
    val cachePath: String,
    /**
     * save user input and generations to prompt cache
     * */
    val promptCacheAll: Boolean = false,
    /**
     * open the prompt cache read-only and do not update it
     * */
    val promptCacheReadOnly: Boolean = false,
)




