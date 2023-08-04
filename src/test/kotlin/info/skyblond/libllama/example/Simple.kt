package info.skyblond.libllama.example

import info.skyblond.libllama.*

/**
 * The simple.cpp
 * */
object Simple {
    init {
        System.setProperty("jna.library.path", "./")
    }

    @JvmStatic
    fun main(args: Array<String>): Unit = with(LibLLaMa.LIB) {
        val prompt = "It's a long argument, but I believe the meaning of life is"
        val contextParameter = getContextParams(
            gqa = 1,
            contextSize = 512,
            batchSize = 1024,
            rmsNormEps = 1e-5f,
            nGpuLayers = 30,
            seed = 12345
        )
        val modelParams = ModelParams(
            modelPath = "/data/llama-model/ggml-llama-2-7b-chat-q8_0.bin"
        )
        val nThread = getProcessorCount()

        println("Seed: ${contextParameter.seed}")

        initLLaMaBackend()
        val (model, ctx) = loadModelAndContextWithParams(contextParameter, modelParams, nThread)

        // tokenize the prompt, here is the token buffer
        val tokensToProcess = tokenize(ctx, prompt, true).toMutableList()

        val ctxSize = llama_n_ctx(ctx)
        // why is context size - 4?
        val maxTokenCount = ctxSize - 4
        check(tokensToProcess.size <= maxTokenCount) {
            "prompt is too long (${tokensToProcess.size} tokens, max $maxTokenCount)"
        }

        // print the initial prompt
        for (token in tokensToProcess) {
            print(llama_token_to_str(ctx, token))
        }

        // keep looping if we haven't run out of context.
        // The LLM keeps a contextual cache memory (aka the context window) of previous token
        // evaluation. Usually, once this cache is full, it is required to recompute a
        // compressed context based on previous tokens, but here we just stop if we run
        // out of context.
        while (llama_get_kv_cache_token_count(ctx) < maxTokenCount) {
            // eval unprocessed tokens
            evalTokens(
                ctx, tokensToProcess,
                // since we're in range of ctx window, we use token cache count as n_past
                llama_get_kv_cache_token_count(ctx),
                nThread
            )
            // clear the input
            tokensToProcess.clear()
            // select best prediction
            val nVocab = llama_n_vocab(ctx)
            val logits = llama_get_logits(ctx).getFloatArray(0, nVocab)

            val candidatesP = llama_token_data_array.ByReference().apply {
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

            // here we select the next token using greedy strategy
            val newTokenId = llama_sample_token_greedy(ctx, candidatesP)
            // test if it's eos
            // in production, we'd better cache this to save some JNA calls
            if (newTokenId == llama_token_eos()) {
                print("[end ot text]")
                break
            }
            // convert token to str and print it
            print(llama_token_to_str(ctx, newTokenId))
            // next loop we will evaluate this new token base on previous nPast tokens
            // and generate another new token
            tokensToProcess.add(newTokenId)
        }

        llama_free(ctx)
        llama_free_model(model)
        llama_backend_free()
    }


}
