package info.skyblond.libllama.example

import info.skyblond.libllama.*

class InferenceCore(
    contextParams: llama_context_params,
    inferenceParams: InferenceParams,
    /**
     * The actual nKeep. [InferenceParams.nKeep] might not be ideal
     * since it might longer than the initial prompt.
     * */
    val nKeep: Int,
) {
    private val ctxSize = contextParams.n_ctx
    private val batchSize = contextParams.n_batch
    private val isInteractive = inferenceParams.interactive
    private val reversePrompts = inferenceParams.reversePrompts

    /**
     * mark the actual context window.
     * unlike ctxSize, which is a computational limitation,
     * nPast is the actual window size when generating a token
     * must not bigger than ctxSize
     * */
    @Volatile
    var nPast: Int = 0
        private set

    /**
     * count how much we still need to generate before we can stop
     * */
    @Volatile
    var nRemain: Int = inferenceParams.nPredict

    /**
     * Test if we can start another round of inference.
     * */
    fun canInference(isReversePrompt: Boolean): Boolean = synchronized(this) {
        // if we still have nRemain, and not found any reverse prompt, we're good to go
        // of if we're in interactive mode, we can always keep going
        nRemain != 0 && !isReversePrompt || isInteractive
    }

    /**
     * infinite text generation via context swapping
     * if we run out of context:
     * - take the nKeep first tokens from the original prompt (from nPast)
     * - take half of the last (ctxSize - n_keep) tokens and recompute the logits in batches
     * */
    private fun contextSwapping(
        tokens: MutableList<Int>,
        lastNTokens: RingTokenBuffer,
    ) = synchronized(this) {
        if (nPast + tokens.size > ctxSize) {
            val nLeft = nPast - nKeep
            // always keep the first token - BOS
            nPast = nKeep.coerceAtLeast(1)
            // insert n_left/2 tokens at the start of embd from last_n_tokens
            tokens.addAll(
                0, lastNTokens.subList(
                    ctxSize - nLeft / 2 - tokens.size,
                    lastNTokens.size - tokens.size
                )
            )
        }
    }

    /**
     * do inference on unprocessed tokens
     * */
    private fun inferenceOnce(
        lib: LibLLaMa, ctx: llama_context, tokens: MutableList<Int>, nThread: Int
    ) = synchronized(this) {
        for (i in tokens.indices step batchSize) {
            val nEval = (tokens.size - i).coerceAtMost(batchSize)
            check( // here we need to control the tokenSize (n_tokens) to nEval
                lib.llama_eval(
                    ctx,
                    tokens.drop(i).take(nEval).toIntArray(), nEval,
                    nPast, nThread
                ) == 0
            ) { "Failed to eval" }
            nPast += nEval
        }
    }

    @Volatile
    private var sessionMark = 0


    fun inference(
        lib: LibLLaMa, lastNTokens: RingTokenBuffer,
        ctx: llama_context, inputTokens: MutableList<Int>,
        sessionTokens: MutableList<Int>, nThread: Int
    ) = synchronized(this) {
        val maxInputSize = ctxSize - 4
        while (inputTokens.isNotEmpty()) {
            val tokensBuffer = ArrayList<Int>(maxInputSize.coerceAtMost(inputTokens.size))
            while (tokensBuffer.size < maxInputSize && inputTokens.isNotEmpty()) {
                tokensBuffer.add(inputTokens.removeFirst())
            }
            contextSwapping(tokensBuffer, lastNTokens)
            // check the session

            if (sessionMark < sessionTokens.size) {
                var i = 0
                while (i < tokensBuffer.size) {
                    // make sure the sessionTokens are the same as tokens
                    if (tokensBuffer[i] != sessionTokens[sessionMark]) {
                        while (sessionTokens.size > sessionMark)
                            sessionTokens.removeLast()
                        break
                    }
                    // accept this cached token
                    nPast++
                    sessionMark++
                    i++
                    // if we used all cached token, exit loop
                    if (sessionMark >= sessionTokens.size) {
                        break
                    }
                }
                // removes the cached tokens from token buffer
                repeat(i) {
                    tokensBuffer.removeFirst()
                }
            }
            inferenceOnce(lib, ctx, tokensBuffer, nThread)
            sessionTokens.addAll(tokensBuffer)
            sessionMark = sessionTokens.size
        }
    }

    fun sample(
        lib: LibLLaMa,
        ctx: llama_context,
        samplingParams: SamplingParams,
        lastNTokens: RingTokenBuffer,
    ): Int = synchronized(this) {
        val tokenId = lib.sampleToken(ctx, samplingParams, lastNTokens)
        nRemain-- // decrease the counter
        tokenId
    }

    fun searchReverseToken(
        lib: LibLLaMa, ctx: llama_context, lastNTokens: RingTokenBuffer
    ): Boolean {
        var result = false
        if (reversePrompts.isNotEmpty()) {
            val lastOutput = StringBuilder()
            for (tokenId in lastNTokens) {
                lastOutput.append(lib.llama_token_to_str(ctx, tokenId))
            }
            // check every reverse prompt and see if it's met
            // If we're not running interactively, the reverse prompt might be tokenized
            // with some following characters, so we'll compensate for that by widening
            // the search window a bit.
            for (reversePrompt in reversePrompts) {
                val extraPadding = if (isInteractive) 0 else 2
                val searchStartPos = if (lastOutput.length > reversePrompt.length + extraPadding) {
                    lastOutput.length - (reversePrompt.length + extraPadding)
                } else 0

                if (lastOutput.indexOf(reversePrompt, searchStartPos) != -1) {
                    result = true
                    break
                }
            }
        }
        return result
    }
}
