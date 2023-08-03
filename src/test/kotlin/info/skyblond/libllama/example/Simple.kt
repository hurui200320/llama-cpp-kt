package info.skyblond.libllama.example

import com.sun.jna.Native
import com.sun.jna.ptr.PointerByReference
import info.skyblond.libllama.LibLLaMa
import info.skyblond.libllama.llama_context
import info.skyblond.libllama.llama_token_data
import info.skyblond.libllama.llama_token_data_array

/**
 * The simple.cpp
 * */
object Simple {

    private val globalCtxPointer: PointerByReference = PointerByReference()
    private lateinit var lib: LibLLaMa

    @JvmStatic
    fun main(args: Array<String>) {
        System.setProperty("jna.library.path", "./")
        lib = Native.load("llama", LibLLaMa::class.java) as LibLLaMa
        Runtime.getRuntime().addShutdownHook(Thread {
            println()
            lib.llama_print_timings(globalCtxPointer.value)
        })

        val params = GPTParams(
            modelPath = "/data/llama-model/ggml-llama-2-7b-chat-q8_0.bin",
            prompt = "I believe the meaning of life is",
            nPredict = -1,
            gqa = 1,
            rmsNormEps = 1e-5f,
            repeatPenalty = 1.8f,
            repeatLastN = 256,
            batchSize = 1024,
            temp = 0.7f,
            topK = 40,
            topP = 0.5f,
            contextSize = 2048,
            seed = 12345,
            nGpuLayers = 30 // TODO: Detect cuBLAS?
        )

        println("Seed: ${params.seed}")
        println("Max devices: ${lib.llama_max_devices()}")

        lib.llama_backend_init(if (params.numa) 1 else 0)

        val (model, ctx) = lib.llama_init_from_gpt_params(params)
        globalCtxPointer.value = ctx!!


        // tokenize the prompt
        var embeddedUserInput = mutableListOf<Int>()
        // Add a space in front of the first character to match OG llama tokenizer behavior
        val promptInput = " " + params.prompt

        embeddedUserInput = llama_tokenize(ctx, promptInput, true).toMutableList()

        val ctxSize = lib.llama_n_ctx(ctx)
        val maxTokenCount = ctxSize - 4
        check(embeddedUserInput.size <= maxTokenCount) {
            "prompt is too long (${embeddedUserInput.size} tokens, max $maxTokenCount)"
        }

        // number of tokens to keep when resetting context
        var nKeep: Int = params.nKeep
        if (params.nKeep < 0 || params.nKeep > embeddedUserInput.size || params.instruct) {
            nKeep = embeddedUserInput.size
        }

        println("\nprompt: '${promptInput}'")
        println("number of tokens in prompt = ${embeddedUserInput.size}")
        for (i in 0 until embeddedUserInput.size) {
            println("%6d -> '%s'".format(embeddedUserInput[i], lib.llama_token_to_str(ctx, embeddedUserInput[i])))
        }

        if (params.nKeep > 0) {
            println("static prompt based on n_keep: '")
            for (i in 0 until params.nKeep) {
                println(lib.llama_token_to_str(ctx, embeddedUserInput[i]))
            }
            println()
        }
        println()

        if (params.antiprompt.isNotEmpty())
            for (s in params.antiprompt)
                println("Reverse prompt: '${s}'")
        if (params.inputPrefixBOS)
            println("Input prefix with BOS")
        if (params.inputPrefix.isNotBlank())
            println("Input prefix: '${params.inputPrefix}'\n")
        if (params.inputSuffix.isNotBlank())
            println("Input suffix: '${params.inputSuffix}'\n")
        println(
            "sampling: repeat_last_n = ${params.repeatLastN}, " +
                    "repeat_penalty = ${params.repeatPenalty}, " +
                    "presence_penalty = ${params.presencePenalty}, " +
                    "frequency_penalty = ${params.frequencyPenalty}, " +
                    "top_k = ${params.topK}, tfs_z = ${params.tfsZ}, top_p = ${params.topP}, " +
                    "typical_p = ${params.typicalP}, temp = ${params.temp}, " +
                    "mirostat = ${params.mirostat}, mirostat_tau = ${params.mirostatTau}, mirostat_eta = ${params.mirostatEta}",
        )
        println(
            "generate: ctxSize = ${ctxSize}, batchSize = ${params.batchSize}," +
                    " nPredict = ${params.nPredict}, n_keep = ${nKeep}"
        )
        println("\n\n")

        // --------------------------


        // print the initial prompt
        for (token in embeddedUserInput) {
            print(lib.llama_token_to_str(ctx, token))
        }

        // if we haven't run out of context
        while (lib.llama_get_kv_cache_token_count(ctx) < maxTokenCount) {
            // eval user input
            check(
                lib.llama_eval(
                    ctx, // context
                    // the input data               the size of input
                    embeddedUserInput.toIntArray(), embeddedUserInput.size,
                    // processed token count, aka the input offset
                    lib.llama_get_kv_cache_token_count(ctx),
                    params.nThread
                ) == 0
            ) { "Failed to eval" }
            // clear the input
            embeddedUserInput.clear()
            // select best prediction
            var newTokenId = 0
            val logits = lib.llama_get_logits(ctx)
            val nVocab = lib.llama_n_vocab(ctx)

            val candidates = llama_token_data().toArray(nVocab)

            candidates.forEachIndexed { index, structure ->
                val obj = structure as llama_token_data
                obj.id = index
                obj.logit = logits.pointer.getFloat(index.toLong() * Float.SIZE_BYTES)
                obj.p = 0.0f
                obj.write()
            }

            val candidatesP = llama_token_data_array.ByReference().apply {
                data = candidates[0].pointer
                size = candidates.size
                sorted = 0
            }

            newTokenId = lib.llama_sample_token_greedy(ctx, candidatesP)

            if (newTokenId == lib.llama_token_eos()) {
                print("[end ot text]")
                break
            }

            print(lib.llama_token_to_str(ctx, newTokenId))
            embeddedUserInput.add(newTokenId)
        }

        lib.llama_free(ctx)
        lib.llama_free_model(model)
        lib.llama_backend_free()
    }

    fun llama_tokenize(ctx: llama_context, text: String, addBOS: Boolean = false): IntArray {
        // the biggest size possible
        val res = IntArray(text.toByteArray().size + if (addBOS) 1 else 0)
        val n = lib.llama_tokenize(ctx, text, res, res.size, if (addBOS) 1 else 0)
        check(n >= 0)

        return res.copyOf(n)
    }


}
