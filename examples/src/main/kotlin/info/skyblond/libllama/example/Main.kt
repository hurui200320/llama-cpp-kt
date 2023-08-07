package info.skyblond.libllama.example

import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference
import info.skyblond.libllama.*
import java.nio.file.Files
import java.nio.file.Path
import java.util.*
import java.util.concurrent.atomic.AtomicReference

/**
 * I tried to implement the main.cpp, but this is the best I got.
 * */
object Main {

    private val globalCtx = AtomicReference<llama_context>()
    private val lib: LibLLaMa
    private val scanner = Scanner(System.`in`)

    /**
     * The flag marking if we are waiting for user input
     * */
    @Volatile
    private var isInteracting: Boolean = false

    init {
        System.setProperty("jna.library.path", "./")
        lib = Native.load("llama", LibLLaMa::class.java) as LibLLaMa
        Runtime.getRuntime().addShutdownHook(Thread {
            globalCtx.get()?.let {
                println()
                lib.llama_print_timings(it)
            }
        })
    }

    @JvmStatic
    fun main(args: Array<String>) {
        val prompt = """
            Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.
            Bob must use prompt "User:" to ask user's request, and must respond with prefix "Bob:".

            User: Hello, Bob.
            Bob: Hello. How may I help you today?
            User: Please tell me the largest city in Europe.
            Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
            User:
        """.trimIndent()
        val contextParameter = lib.getContextParams(
            gqa = 1,
            contextSize = 512,
            batchSize = 1024,
            rmsNormEps = 1e-5f,
            nGpuLayers = 10,
//            seed = 12345
        ).also { println("Seed: ${it.seed}") }
        val modelParams = ModelParams(
            modelPath = "/data/llama-model/ggml-llama-2-7b-chat-f16.bin"
        )
        val guidanceParams = GuidanceParams(
            negativePrompt = "",
            scale = 1.0f
        )
        val inferenceParams = InferenceParams(
            nKeep = 72,
            reversePrompts = mutableListOf("User:"),
            inputPrefix = " ",
            interactive = true
        )
        val samplingParams = SamplingParams(
            temp = 0.7f,
            topK = 40,
            topP = 0.5f,
            repeatPenalty = 1.22f,
            repeatLastN = 256,
        )
        val persistenceParams = PersistenceParams(
            cachePath = "",
            promptCacheAll = true,
        )
        val nThread = getProcessorCount()

        // end of parameters, start doing work

        lib.initLLaMaBackend()
        val (model, ctx) = lib.loadModelAndContextWithParams(contextParameter, modelParams, nThread)
        globalCtx.set(ctx!!)

        // the guidance context
        val guidanceCtx = if (guidanceParams.scale > 1.0f) {
            lib.llama_new_context_with_model(model, contextParameter)
        } else null

        // load status
        val sessionTokens = if (persistenceParams.cachePath.isNotEmpty()
            && Files.exists(Path.of(persistenceParams.cachePath))
        ) {
            val data = IntArray(contextParameter.n_ctx)
            val nTokenCountOut = IntByReference(0)

            check(
                lib.llama_load_session_file(
                    ctx, persistenceParams.cachePath,
                    data, contextParameter.n_ctx, nTokenCountOut
                ).toInt() == 1
            ) {
                "failed to load session file '${persistenceParams.cachePath}'"
            }
            lib.llama_set_rng_seed(ctx, contextParameter.seed)
            println("Loaded ${nTokenCountOut.value} tokens from session file '${persistenceParams.cachePath}'")
            data.copyOf(nTokenCountOut.value).toMutableList()
        } else mutableListOf()

        // tokenize the input prompt
        val promptProcessedText = " $prompt"
        // tokens of the prompt
        // we tokenize the prompt if any of them is true:
        // + we need to process user input soon (interactiveFirst)
        // + we need instructions from user (alpacaInstruct)
        // + we have some prompt to process (textInput not empty)
        // + we don't loaded from file
        val promptTokenized = if (
            inferenceParams.interactiveFirst
            || inferenceParams.alpacaInstruct
            || promptProcessedText.isNotEmpty()
            || sessionTokens.isEmpty()
        ) {
            lib.tokenize(ctx, promptProcessedText, true).toMutableList()
        } else { // otherwise we use the loaded status
            // note: we cannot reuse the list, we need copy it
            sessionTokens.toIntArray().toMutableList()
        }

        // Tokenize negative prompt
        val guidanceInputToken = mutableListOf<Int>()
        var guidanceOffset = 0
        var originalPromptLength = 0
        if (guidanceCtx != null) {
            guidanceInputToken.addAll(lib.tokenize(ctx, " ${guidanceParams.negativePrompt}", true).toList())
            val originalInput = lib.tokenize(ctx, promptProcessedText, true)
            originalPromptLength = originalInput.size
            guidanceOffset = guidanceInputToken.size - originalPromptLength
        }

        val ctxSize = lib.llama_n_ctx(ctx)
        // TODO how to take care of super long token?
        //      eval multiple times?
        check(promptTokenized.size <= ctxSize - 4) {
            "prompt is too long (${promptTokenized.size} tokens, max ${ctxSize - 4})"
        }

        // test the similarity of loaded session with current setup
        var nMatchingSessionTokens = 0
        if (sessionTokens.isNotEmpty()) {
            // loop and count:
            // if we have already finished the promptTokenized, then break
            // if the tokenId in loaded session does not match the input tokens
            //      (may come from different mode that don't use loaded session), break
            // otherwise count it as matched
            for (tokenId in sessionTokens) {
                if (nMatchingSessionTokens >= promptTokenized.size || tokenId != promptTokenized[nMatchingSessionTokens])
                    break
                nMatchingSessionTokens++
            }

            if (promptProcessedText.isEmpty() && nMatchingSessionTokens == promptTokenized.size) {
                println("Using full prompt from session file")
            } else if (nMatchingSessionTokens >= promptTokenized.size) {
                // note: nMatchingSessionTokens will equal to promptTokenized.size
                // since we break as soon as they are equal
                println("Session file is exactly match for prompt")
            } else if (nMatchingSessionTokens < promptTokenized.size / 2) {
                println("warning: session file has low similarity to prompt (${nMatchingSessionTokens} / ${promptTokenized.size} tokens); will mostly be reevaluated")
            } else {
                println("session file matches $nMatchingSessionTokens / ${promptTokenized.size} tokens of prompt")
            }
        }

        // if the cache is longer than the full tokenized prompt,
        // force reevaluation of the last token to recalculate the cached logits
        if (promptTokenized.isNotEmpty()
            && nMatchingSessionTokens >= promptTokenized.size
            && sessionTokens.size > promptTokenized.size
        ) {
            while (sessionTokens.size > promptTokenized.size - 1)
                sessionTokens.removeLast()
        }


        // calculate the actual nKeep based on parameters and current setup
        val nKeep = if (inferenceParams.nKeep !in 0..promptTokenized.size || inferenceParams.alpacaInstruct) {
            promptTokenized.size // in this case, we reset the nKeep to actual input size
        } else inferenceParams.nKeep

        // tokenize prefix & suffix for alpaca
        val alpacaInputPrefixToken = lib.tokenize(ctx, "\n\n### Instruction:\n\n", true)
        val alpacaInputSuffixToken = lib.tokenize(ctx, "\n\n### Response:\n\n", false)

        // in instruct mode, we inject a prefix and a suffix to each input by the user
        if (inferenceParams.alpacaInstruct) {
            inferenceParams.reversePrompts.add("### Instruction:\n\n")
        }

        // print some info
        println()
        println("Actual prompt: $promptProcessedText")
        println("Token count of the prompt: ${promptTokenized.size}")
        for (i in promptTokenized.indices) {
            println("%6d -> '%s'".format(promptTokenized[i], lib.llama_token_to_str(ctx, promptTokenized[i])))
        }

        if (guidanceCtx != null) {
            println()
            println("Negative prompt: ${guidanceParams.negativePrompt}")
            println("Token count of the negative prompt: ${guidanceInputToken.size}")
            for (i in guidanceInputToken.indices) {
                println("%6d -> '%s'".format(guidanceInputToken[i], lib.llama_token_to_str(ctx, guidanceInputToken[i])))
            }
        }

        if (nKeep > 0) {
            println()
            print("Static prompt based on nKeep = $nKeep: '")
            for (i in 0 until nKeep) {
                print(lib.llama_token_to_str(ctx, promptTokenized[i]))
            }
            println("'")
        }

        if (inferenceParams.interactive) {
            println("interactive mode on.")
            if (inferenceParams.reversePrompts.isNotEmpty()) {
                for (p in inferenceParams.reversePrompts) {
                    println("Reverse prompt: '${p}'")
                }
            }
            if (inferenceParams.inputPrefixBOS) {
                println("Input prefix with BOS")
            }
            if (inferenceParams.inputPrefix.isNotEmpty()) {
                println("Input prefix: '${inferenceParams.inputPrefix}'")
            }
            if (inferenceParams.inputSuffix.isNotEmpty()) {
                println("Input suffix: '${inferenceParams.inputSuffix}'")
            }
        }

        println()
        println(contextParameter)
        println(samplingParams)
        println(inferenceParams)
        println("\n")

        // TODO: grammar
//        grammar_parser::parse_state parsed_grammar;
//        llama_grammar *             grammar = NULL;
//        if (!params.grammar.empty()) {
//            parsed_grammar = grammar_parser::parse(params.grammar.c_str());
//            // will be empty (default) if there are parse errors
//            if (parsed_grammar.rules.empty()) {
//                return 1;
//            }
//            fprintf(stderr, "%s: grammar:\n", __func__);
//            grammar_parser::print_grammar(stderr, parsed_grammar);
//            fprintf(stderr, "\n");
//
//            {
//                auto it = params.logit_bias.find(llama_token_eos());
//                if (it != params.logit_bias.end() && it->second == -INFINITY) {
//                fprintf(stderr,
//                    "%s: warning: EOS token is disabled, which will cause most grammars to fail\n", __func__);
//            }
//            }
//
//            std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
//            grammar = llama_grammar_init(
//                grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
//        }


        val lastNTokens = RingTokenBuffer(ctxSize)

        var isReversePrompt = false
        var inputEcho = false

        if (inferenceParams.interactive) {
            println(" - Press Return to return control to LLaMa.")
            println(" - To return control without starting a new line, end your input with '/'.")
            println(" - If you want to submit another line, end your input with '\\'.")
            println("== Running in interactive mode. ==")
            isInteracting = inferenceParams.interactiveFirst
        }

        // mark the actual context window.
        // unlike ctxSize, which is a computational limitation,
        // nPast is the actual window size when generating a token
        // must not bigger than ctxSize
        var nPast = 0
        // count how much we still need to generate before we can stop
        var nRemain = inferenceParams.nPredict
        // mark the total token fed into llama
        var nConsumed = 0
        // mark the token from session fed into llama
        var nSessionConsumed = 0
        // context window for guidance prompt
        var nPastGuidance = 0

        // the buffer of unprocessed tokens
        val tokens = mutableListOf<Int>()
        val tokensGuidance = mutableListOf<Int>()

        // warm up and reset the timing
        run {
            val temp = intArrayOf(lib.llama_token_bos())
            lib.evalTokens(ctx, temp, 0, nThread)
            lib.llama_reset_timings(ctx)
        }

        // print the prompt
        print(promptProcessedText)

        // the main loop
        while (nRemain != 0 && !isReversePrompt || inferenceParams.interactive) {
            // predict next token
            if (tokens.isNotEmpty()) {
                // Note: ctxSize - 4 here is to match the logic for commandline prompt
                // handling via --prompt or --file which uses the same value.
                val maxInputSize = ctxSize - 4 // otherwise the model will forget the earlier input
                // Ensure the input doesn't exceed the context size by truncating embd if necessary.
                // TODO: maybe alert user but not truncating, just use it as it.
                if (tokens.size > maxInputSize) {
                    System.err.println("<<Input too long, skiping ${tokens.size - maxInputSize} tokens>>")
                    while (tokens.size > maxInputSize)
                        tokens.removeLast()
                }

                // infinite text generation via context swapping
                // if we run out of context:
                // - take the nKeep first tokens from the original prompt (from nPast)
                // - take half of the last (ctxSize - n_keep) tokens and recompute the logits in batches
                if (nPast + tokens.size + guidanceOffset.coerceAtLeast(0) > ctxSize) {
                    val nLeft = nPast - nKeep
                    // always keep the first token - BOS
                    nPast = nKeep.coerceAtLeast(1)
                    nPastGuidance = (nKeep + guidanceOffset).coerceAtLeast(1)
                    // insert n_left/2 tokens at the start of embd from last_n_tokens
                    tokens.addAll(
                        0, lastNTokens.subList(
                            ctxSize - nLeft / 2 - tokens.size,
                            lastNTokens.size - tokens.size
                        )
                    )
                    // stop saving session if we run out of context
                    // TODO: Why?
                }

                // try to reuse a matching prefix from the loaded session instead of
                // re-eval (via n_past)
                if (nSessionConsumed < sessionTokens.size) {
                    var i = 0
                    while (i < tokens.size) {
                        // make sure the sessionTokens are the same as tokens
                        if (tokens[i] != sessionTokens[nSessionConsumed]) {
                            while (sessionTokens.size > nSessionConsumed)
                                sessionTokens.removeLast()
                            break
                        }
                        // accept this cached token
                        nPast++
                        nSessionConsumed++
                        i++
                        // if we used all cached token, exit loop
                        if (nSessionConsumed >= sessionTokens.size) {
                            break
                        }
                    }
                    // removes the cached tokens from token buffer
                    repeat(i) {
                        tokens.removeFirst()
                    }
                }

                // evaluate tokens in batches
                // token buffer is typically prepared beforehand to fit within a batch,
                // but not always
                if (guidanceCtx != null) {
                    // if we didn't processed all guidance input,
                    // process it first
                    val inputBuffer = if (nPastGuidance < guidanceInputToken.size) {
                        // Guidance context should have the same data with these modifications:
                        // * Replace the initial prompt
                        // * Shift everything by guidance_offset
                        tokensGuidance.clear()
                        tokensGuidance.addAll(guidanceInputToken)
                        // if unprocessed tokens are longer than original,
                        // add the newer token into guidance buffer.
                        // otherwise the llama will forget the guidance
                        if (tokens.size > originalPromptLength) {
                            tokensGuidance.addAll(tokens.subList(originalPromptLength, tokens.size))
                        }
                        // we use our newly created guidance as input
                        tokensGuidance.toIntArray()
                    } else {
                        // just use token
                        tokens.toIntArray()
                    }

                    // do inference in batch
                    for (i in inputBuffer.indices step contextParameter.n_batch) {
                        // each time process at most batchSize tokens
                        val nEval = (inputBuffer.size - i).coerceAtMost(contextParameter.n_batch)
                        check( // here we need to control the tokenSize (n_tokens) to nEval
                            lib.llama_eval(
                                guidanceCtx,
                                inputBuffer.drop(i).take(nEval).toIntArray(), nEval,
                                nPastGuidance, nThread
                            ) == 0
                        ) { "Failed to eval guidance" }
                        nPastGuidance += nEval
                    }
                }

                // do inference on unprocessed tokens
                for (i in tokens.indices step contextParameter.n_batch) {
                    val nEval = (tokens.size - i).coerceAtMost(contextParameter.n_batch)
                    check( // here we need to control the tokenSize (n_tokens) to nEval
                        lib.llama_eval(
                            ctx,
                            tokens.drop(i).take(nEval).toIntArray(), nEval,
                            nPast, nThread
                        ) == 0
                    ) { "Failed to eval" }
                    nPast += nEval
                }

                // update the session cache
                if (tokens.isNotEmpty() && persistenceParams.cachePath.isNotEmpty()) {
                    sessionTokens.addAll(tokens)
                    nSessionConsumed = sessionTokens.size
                }
            }

            // now we processed all unprocessed tokens, clear the buffer
            tokens.clear()
            tokensGuidance.clear()

            // if we consumed all input prompt, and we're not waiting for user input
            // sample the next token
            if (promptTokenized.size <= nConsumed && !isInteracting) {
                val tokenId = lib.sampleToken(ctx, samplingParams, lastNTokens, guidanceCtx, guidanceParams.scale)
                // add it to the context
                tokens.add(tokenId)
                // echo this to console
                inputEcho = true
                // decrement remaining sampling budget
                nRemain--
            } else {
                // some user input remains from prompt or interaction, forward it to processing
                while (promptTokenized.size > nConsumed) {
                    tokens.add(promptTokenized[nConsumed])
                    lastNTokens.removeFirst()
                    lastNTokens.add(promptTokenized[nConsumed])
                    nConsumed++
                    // stop if unprocessed token size reached batch size
                    // we will leave it to next loop
                    if (tokens.size >= contextParameter.n_batch) {
                        break
                    }
                }
            }

            // display text
            if (inputEcho) {
                for (tokenId in tokens) {
                    print(lib.llama_token_to_str(ctx, tokenId))
                }
            }

            // if we processed all user input
            if (promptTokenized.size <= nConsumed) {
                // check for reverse prompt
                if (inferenceParams.reversePrompts.isNotEmpty()) {
                    val lastOutput = StringBuilder()
                    for (tokenId in lastNTokens) {
                        lastOutput.append(lib.llama_token_to_str(ctx, tokenId))
                    }
                    isReversePrompt = false
                    // check every reverse prompt and see if it's met
                    // If we're not running interactively, the reverse prompt might be tokenized
                    // with some following characters, so we'll compensate for that by widening
                    // the search window a bit.
                    for (reversePrompt in inferenceParams.reversePrompts) {
                        val extraPadding = if (inferenceParams.interactive) 0 else 2
                        val searchStartPos = if (lastOutput.length > reversePrompt.length + extraPadding) {
                            lastOutput.length - (reversePrompt.length + extraPadding)
                        } else 0

                        if (lastOutput.indexOf(reversePrompt, searchStartPos) != -1) {
                            if (inferenceParams.interactive) {
                                isInteracting = true // now we're waiting for user input
                            }
                            isReversePrompt = true
                            break
                        }
                    }
                }

                // deal with end of text
                if (lastNTokens.last() == lib.llama_token_eos()) {
                    if (inferenceParams.interactive) {
                        // for interactive mode, treat as a reverse prompt
                        if (inferenceParams.reversePrompts.isNotEmpty()) {
                            // tokenize and inject first reverse prompt
                            val firstReversePrompt = lib.tokenize(ctx, inferenceParams.reversePrompts.first(), false)
                            promptTokenized.addAll(firstReversePrompt.toList())
                            isReversePrompt = true
                        }
                        isInteracting = true
                        println()
                    } else if (inferenceParams.alpacaInstruct) {
                        // for alpaca instruct mode, we're just waiting for next instruction
                        isInteracting = true
                    }
                }

                // waiting for user input
                if (nPast > 0 && isInteracting) {
                    if (inferenceParams.alpacaInstruct) print("\n> ")
                    if (inferenceParams.inputPrefixBOS) promptTokenized.add(lib.llama_token_bos())
                    val buffer = StringBuilder()
                    if (inferenceParams.inputPrefix.isNotEmpty()) {
                        buffer.append(inferenceParams.inputPrefix)
                        print(buffer.toString())
                    }
                    // asking user input
                    var line: String
                    var anotherLine: Boolean
                    do {
                        line = scanner.nextLine()
                        anotherLine = line.trimEnd().lastOrNull() == '\\'
                        if (anotherLine) {
                            buffer.append(line.trimEnd().removeSuffix("\\"))
                                .append("\n")
                        } else {
                            buffer.append(line)
                        }
                    } while (anotherLine)

                    // check empty input for just return control to keep generating text
                    if (buffer.length > 1) {
                        // append input suffix if any
                        if (inferenceParams.inputSuffix.isNotEmpty()) {
                            buffer.append(inferenceParams.inputSuffix)
                            print(inferenceParams.inputSuffix)
                        }
                        // instruct mode: insert instruction prefix
                        if (inferenceParams.alpacaInstruct && !isReversePrompt) {
                            nConsumed = promptTokenized.size
                            promptTokenized.addAll(alpacaInputPrefixToken.toList())
                        }
                        val bufferTokenized = lib.tokenize(ctx, buffer.toString(), false)
                        promptTokenized.addAll(bufferTokenized.toList())
                        // instruct mode: insert response suffix
                        if (inferenceParams.alpacaInstruct) {
                            promptTokenized.addAll(alpacaInputSuffixToken.toList())
                        }
                        nRemain -= bufferTokenized.size
                    }
                    // do not echo user input
                    inputEcho = false
                }

                if (nPast > 0) {
                    // reset grammar state if we're restarting generation
                    // TODO grammar
//                    if (isInteracting) {
                    // if (grammar != NULL) {
                    //     llama_grammar_free(grammar);
                    //
                    //     std::vector<const llama_grammar_element *> grammar_rules(
                    //         parsed_grammar.c_rules());
                    //     grammar = llama_grammar_init(
                    //         grammar_rules.data(), grammar_rules.size(),
                    //         parsed_grammar.symbol_ids.at("root"));
                    // }
//                    }
                    isInteracting = false
                }
            }

            // end of text token for none interactive mode
            if (tokens.isNotEmpty()
                && tokens.last() == lib.llama_token_eos()
                && !inferenceParams.alpacaInstruct
                && !inferenceParams.interactive
            ) {
                println(" [end of text]")
                break
            }

            // In interactive mode, respect the maximum number of tokens and drop back to user
            // input when reached.
            if (inferenceParams.interactive && nRemain <= 0 && inferenceParams.nPredict != -1) {
                nRemain = inferenceParams.nPredict
                isInteracting = true
            }
        }

        // save session
        if (persistenceParams.cachePath.isNotEmpty()
            && persistenceParams.promptCacheAll
            && !persistenceParams.promptCacheReadOnly
        ) {
            println("saving final output to session file '${persistenceParams.cachePath}'")
            lib.llama_save_session_file(
                ctx, persistenceParams.cachePath,
                sessionTokens.toIntArray(), sessionTokens.size
            )
        }

        lib.llama_print_timings(ctx)
        if (guidanceCtx != null)
            lib.llama_free(guidanceCtx)
        globalCtx.set(null)
        lib.llama_free(ctx)
        lib.llama_free_model(model)
        lib.llama_backend_free()
    }
}
