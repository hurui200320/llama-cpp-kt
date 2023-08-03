package info.skyblond.libllama

import com.sun.jna.Library
import com.sun.jna.ptr.ByteByReference
import com.sun.jna.ptr.FloatByReference
import com.sun.jna.ptr.IntByReference

@Suppress("FunctionName", "MemberVisibilityCanBePrivate", "unused")
interface LibLLaMa : Library {
    @Suppress("unused")
    companion object {
        const val LLAMA_FILE_MAGIC_GGJT = 0x67676a74u
        const val LLAMA_FILE_MAGIC_GGLA = 0x67676c61u
        const val LLAMA_FILE_MAGIC_GGMF = 0x67676d66u
        const val LLAMA_FILE_MAGIC_GGML = 0x67676d6cu
        const val LLAMA_FILE_MAGIC_GGSN = 0x6767736eu

        const val LLAMA_FILE_VERSION = 3
        const val LLAMA_FILE_MAGIC = LLAMA_FILE_MAGIC_GGJT
        const val LLAMA_FILE_MAGIC_UNVERSIONED = LLAMA_FILE_MAGIC_GGML
        const val LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
        const val LLAMA_SESSION_VERSION = 1

        const val LLAMA_DEFAULT_SEED = 0xFFFFFFFF

        const val LLAMA_DEFAULT_RMS_EPS = 5e-6f
    }

    fun llama_max_devices(): Int

    fun llama_model_quantize_default_params(): llama_model_quantize_params.ByValue
    fun llama_context_default_params(): llama_context_params.ByValue

    /**
     * Return 0: false; return 1: true.
     * */
    fun llama_mmap_supported(): Byte
    fun llama_mlock_supported(): Byte

    /**
     * Initialize the llama + ggml backend
     * If numa is 1, use NUMA optimizations
     * Call once at the start of the program
     * */
    fun llama_backend_init(numa: Byte)

    /**
     * Call once at the end of the program - currently only used for MPI
     * */
    fun llama_backend_free()

    fun llama_time_us(): Long

    /**
     * Various functions for loading a ggml llama model.
     * Allocate (almost) all memory needed for the model.
     * */
    fun llama_load_model_from_file(
        path_model: String, params: llama_context_params.ByValue
    ): llama_model

    fun llama_free_model(model: llama_model)


    fun llama_new_context_with_model(
        model: llama_model, params: llama_context_params.ByValue
    ): llama_context

    /**
     * Frees all allocated memory
     */
    fun llama_free(ctx: llama_context)

    /**
     * Returns 0 on success
     */
    fun llama_model_quantize(
        fname_inp: String,
        fname_out: String,
        params: llama_model_quantize_params.ByReference
    ): Int

    /**
     * Apply a LoRA adapter to a loaded model
     * path_base_model is the path to a higher quality model to use as a base for
     * the layers modified by the adapter. Can be NULL to use the current loaded model.
     * The model needs to be reloaded before applying a new adapter, otherwise the adapter
     * will be applied on top of the previous one
     * Returns 0 on success
     * */
    fun llama_model_apply_lora_from_file(
        model: llama_model,
        path_lora: String,
        path_base_model: String?,
        n_threads: Int
    ): Int

    /**
     * Returns the number of tokens in the KV cache
     * */
    fun llama_get_kv_cache_token_count(ctx: llama_context): Int

    /**
     * Sets the current rng seed.
     * */
    fun llama_set_rng_seed(ctx: llama_context, seed: Int)

    /**
     * Returns the maximum size in bytes of the state (rng, logits, embedding
     * and kv_cache) - will often be smaller after compacting tokens
     * */
    fun llama_get_state_size(ctx: llama_context): Int

    /**
     * Copies the state to the specified destination address.
     * Destination needs to have allocated enough memory.
     * Returns the number of bytes copied
     * */
    fun llama_copy_state_data(ctx: llama_context, dst: ByteByReference): Int

    /**
     * Set the state reading from the specified address
     * Returns the number of bytes read
     * */
    fun llama_set_state_data(ctx: llama_context, src: ByteByReference): Int

    /**
     * Save/load session file.
     * Return 1: true; return 0: false.
     * */
    fun llama_load_session_file(
        ctx: llama_context, path_session: String, tokens_out: IntArray,
        n_token_capacity: Int, n_token_count_out: IntByReference
    ): Byte

    fun llama_save_session_file(
        ctx: llama_context, path_session: String, tokens: IntByReference,
        n_token_count: Int
    )

    /**
     * Run the llama inference to obtain the logits and probabilities for the next token.
     * tokens + n_tokens is the provided batch of new tokens to process
     * n_past is the number of tokens to use from previous eval calls
     * Returns 0 on success
     * */
    fun llama_eval(
        ctx: llama_context, tokens: IntArray,
        n_tokens: Int, n_past: Int, n_threads: Int
    ): Int

    /**
     * Same as llama_eval, but use float matrix input directly.
     * */
    fun llama_eval_embd(
        ctx: llama_context, embd: FloatByReference,
        n_tokens: Int, n_past: Int, n_threads: Int
    ): Int

    /**
     * Export a static computation graph for context of 511 and batch size of 1
     * NOTE: since this functionality is mostly for debugging and demonstration purposes, we hardcode these
     *       parameters here to keep things simple
     * IMPORTANT: do not use for anything else other than debugging and testing!
     * */
    fun llama_eval_export(ctx: llama_context, fname: String): Int

    /**
     * Convert the provided text into tokens.
     * The tokens pointer must be large enough to hold the resulting tokens.
     * Returns the number of tokens on success, no more than n_max_tokens
     * Returns a negative number on failure - the number of tokens that would have been returned.
     * add_bos: 1 means add, 0 means don't.
     * TODO: not sure if correct
     * */
    fun llama_tokenize(
        ctx: llama_context, text: String,
        tokens: IntArray, n_max_tokens: Int,
        add_bos: Byte
    ): Int

    fun llama_tokenize_with_model(
        model: llama_model, text: String,
        tokens: IntByReference, n_max_tokens: Int,
        add_bos: Byte
    ): Int

    fun llama_n_vocab(ctx: llama_context): Int
    fun llama_n_ctx(ctx: llama_context): Int
    fun llama_n_embd(ctx: llama_context): Int

    fun llama_n_vocab_from_model(model: llama_model): Int
    fun llama_n_ctx_from_model(model: llama_model): Int
    fun llama_n_embd_from_model(model: llama_model): Int

    /**
     * Get the vocabulary as output parameters.
     * Returns number of results.
     * */
    fun llama_get_vocab(
        ctx: llama_context, strings: Array<String>,
        scores: FloatByReference, capacity: Int
    ): Int

    fun llama_get_vocab_from_model(
        model: llama_model, strings: Array<String>,
        scores: FloatArray, capacity: Int
    ): Int

    /**
     * Token logits obtained from the last call to llama_eval()
     * The logits for the last token are stored in the last row
     * Can be mutated in order to change the probabilities of the next token
     * Rows: n_tokens
     * Cols: n_vocab
     * */
    fun llama_get_logits(ctx: llama_context): FloatByReference

    /**
     * Get the embeddings for the input
     * shape: n_embd (1-dimensional)
     * */
    fun llama_get_embeddings(ctx: llama_context): FloatByReference

    /**
     * Token Id -> String. Uses the vocabulary in the provided context
     * */
    fun llama_token_to_str(ctx: llama_context, token: Int): String

    fun llama_token_to_str_with_model(model: llama_model, token: Int): String


    /**
     * Special tokens: beginning-of-sentence
     */
    fun llama_token_bos(): Int

    /**
     * Special tokens: end-of-sentence
     */
    fun llama_token_eos(): Int

    /**
     * Special tokens: next-line
     */
    fun llama_token_nl(): Int

    fun llama_grammar_init(
        rules: Array<llama_grammar_element.ByReference>,
        n_rules: Int, start_rule_index: Int
    ): llama_grammar

    fun llama_grammar_free(grammar: llama_grammar)

    /**
     * Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
     * */
    fun llama_sample_repetition_penalty(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        last_tokens: IntByReference, last_tokens_size: Int,
        penalty: Float
    )

    /**
     * Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
     * */
    fun llama_sample_frequency_and_presence_penalties(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        last_tokens: IntByReference, last_tokens_size: Int,
        alpha_frequency: Float, alpha_presence: Float
    )

    /**
     * Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
     *
     * @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
     * @param guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
     * @param scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
     * */
    fun llama_sample_classifier_free_guidance(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        guidance_ctx: llama_context, scale: Float
    )

    /**
     * Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
     * */
    fun llama_sample_softmax(
        ctx: llama_context, candidates: llama_token_data_array.ByReference
    )

    /**
     * Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
     * */
    fun llama_sample_top_k(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        k: Int, min_keep: Int
    )

    /**
     * Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
     * */
    fun llama_sample_top_p(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        p: Float, min_keep: Int
    )

    /**
     * Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
     * */
    fun llama_sample_tail_free(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        z: Float, min_keep: Int
    )

    /**
     * Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
     * */
    fun llama_sample_typical(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        p: Float, min_keep: Int
    )

    fun llama_sample_temperature(
        ctx: llama_context, candidates: llama_token_data_array.ByReference, temp: Float
    )

    /**
     * Apply constraints from grammar
     * */
    fun llama_sample_grammar(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        grammar: llama_grammar
    )

    /**
     * Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
     * @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
     * @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
     * @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
     * @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
     * @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
     * */
    fun llama_sample_token_mirostat(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        tau: Float, eta: Float, m: Int, mu: FloatByReference
    ): Int

    /**
     * Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
     *
     * @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
     * @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
     * @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
     * @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
     * */
    fun llama_sample_token_mirostat_v2(
        ctx: llama_context, candidates: llama_token_data_array.ByReference,
        tau: Float, eta: Float, mu: FloatByReference
    ): Int

    /**
     * Selects the token with the highest probability.
     * */
    fun llama_sample_token_greedy(
        ctx: llama_context, candidates: llama_token_data_array.ByReference
    ): Int

    /**
     * Randomly selects a token from the candidates based on their probabilities.
     * */
    fun llama_sample_token(ctx: llama_context, candidates: llama_token_data_array.ByReference): Int

    /**
     * Accepts the sampled token into the grammar
     * */
    fun llama_grammar_accept_token(
        ctx: llama_context, grammar: llama_grammar, token: Int
    )

    fun llama_get_timings(ctx: llama_context): llama_timings.ByValue
    fun llama_print_timings(ctx: llama_context)
    fun llama_reset_timings(ctx: llama_context)

    fun llama_print_system_info(): String
}