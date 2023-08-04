package info.skyblond.libllama.example

import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference
import info.skyblond.libllama.*
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicReference

/**
 * I tried to implement the main.cpp, but this is the best I got.
 * */
object Main {

    private val globalCtx = AtomicReference<llama_context>()
    private val lib: LibLLaMa

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
        val prompt = "I believe the meaning of life is"
        val contextParameter = lib.getContextParams(
            gqa = 1,
            contextSize = 512,
            batchSize = 1024,
            rmsNormEps = 1e-5f,
            nGpuLayers = 30,
            seed = 12345
        ).also { println("Seed: ${it.seed}") }
        val modelParams = ModelParams(
            modelPath = "/data/llama-model/ggml-llama-2-7b-chat-q8_0.bin"
        )
        val guidanceParams = GuidanceParams(
            negativePrompt = "",
            scale = 1.0f
        )
        val inferenceParams = InferenceParams(
            nKeep = 64
        )
        val samplingParams = SamplingParams(
            temp = 0.7f,
            topK = 40,
            topP = 0.5f,
            repeatPenalty = 1.18f,
            repeatLastN = 256,
        )
        val persistenceParams = PersistenceParams(
            cachePath = "/data/llama/cache",
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
        val sessionTokens = if (Files.exists(Path.of(persistenceParams.cachePath))) {
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
            data.toMutableList()
        } else mutableListOf()

        // tokenize the input prompt
        var promptProcessedText = " $prompt"
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
            lib.tokenize(ctx, promptProcessedText, true)
        } else { // otherwise we use the loaded status
            sessionTokens.toIntArray()
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
            // if we have already finished the inputTokens, then break
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
                // note: nMatchingSessionTokens will equal to inputTokens.size
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

        // determine newline token
        val tokenNewline = lib.tokenize(ctx, "\n", false)

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
        var needToSaveSession =
            persistenceParams.cachePath.isNotEmpty() && nMatchingSessionTokens < promptTokenized.size

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
                // if (n_session_consumed < (int) sessionTokens.size()) {
                //     size_t i = 0;
                //     for ( ; i < tokens.size(); i++) {
                //         if (embd[i] != sessionTokens[n_session_consumed]) {
                //             sessionTokens.resize(n_session_consumed);
                //             break;
                //         }
                //
                //         n_past++;
                //         n_session_consumed++;
                //
                //         if (n_session_consumed >= (int) sessionTokens.size()) {
                //             ++i;
                //             break;
                //         }
                //     }
                //     if (i > 0) {
                //         tokens.erase(tokens.begin(), tokens.begin() + i);
                //     }
                // }
                //
                // // evaluate tokens in batches
                // // embd is typically prepared beforehand to fit within a batch, but not always
                //
                // if (guidanceCtx) {
                //     int input_size = 0;
                //     llama_token* input_buf = NULL;
                //
                //     if (n_past_guidance < (int) guidance_inp.size()) {
                //         // Guidance context should have the same data with these modifications:
                //         //
                //         // * Replace the initial prompt
                //         // * Shift everything by guidance_offset
                //         embd_guidance = guidance_inp;
                //         if (tokens.begin() + original_prompt_len < tokens.end()) {
                //             tokensGuidance.insert(
                //                 tokensGuidance.end(),
                //                 tokens.begin() + original_prompt_len,
                //                 tokens.end()
                //             );
                //         }
                //
                //         input_buf = tokensGuidance.data();
                //         input_size = tokensGuidance.size();
                //         //fprintf(stderr, "\n---------------------\n");
                //         //for (int i = 0; i < (int) tokensGuidance.size(); i++) {
                //             //fprintf(stderr, "%s", llama_token_to_str(ctx, embd_guidance[i]));
                //         //}
                //         //fprintf(stderr, "\n---------------------\n");
                //     } else {
                //         input_buf = tokens.data();
                //         input_size = tokens.size();
                //     }
                //
                //     for (int i = 0; i < input_size; i += params.n_batch) {
                //         int n_eval = std::min(input_size - i, params.n_batch);
                //         if (llama_eval(guidanceCtx, input_buf + i, n_eval, n_past_guidance, params.n_threads)) {
                //             fprintf(stderr, "%s : failed to eval\n", __func__);
                //             return 1;
                //         }
                //
                //         n_past_guidance += n_eval;
                //     }
                // }
                //
                // for (int i = 0; i < (int) tokens.size(); i += params.n_batch) {
                //     int n_eval = (int) tokens.size() - i;
                //     if (n_eval > params.n_batch) {
                //         n_eval = params.n_batch;
                //     }
                //     if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                //         fprintf(stderr, "%s : failed to eval\n", __func__);
                //         return 1;
                //     }
                //     n_past += n_eval;
                // }
                //
                // if (tokens.size() > 0 && !path_session.empty()) {
                //     sessionTokens.insert(sessionTokens.end(), tokens.begin(), tokens.end());
                //     n_session_consumed = sessionTokens.size();
                // }
            }
            /*

            tokens.clear();
            tokensGuidance.clear();

            if ((int) inputTokens.size() <= n_consumed && !is_interacting) {
                llama_token id = lib.sampleToken()

                // add it to the context
                tokens.push_back(id);

                // echo this to console
                input_echo = true;

                // decrement remaining sampling budget
                --n_remain;
            } else {
                // some user input remains from prompt or interaction, forward it to processing
                while ((int) inputTokens.size() > n_consumed) {
                    tokens.push_back(embd_inp[n_consumed]);
                    last_n_tokens.erase(last_n_tokens.begin());
                    last_n_tokens.push_back(embd_inp[n_consumed]);
                    ++n_consumed;
                    if ((int) tokens.size() >= params.n_batch) {
                        break;
                    }
                }
            }

            // display text
            if (input_echo) {
                for (auto id : tokens) {
                    printf("%s", llama_token_to_str(ctx, id));
                }
                fflush(stdout);
            }
            // reset color to default if we there is no pending user input
            if (input_echo && (int)inputTokens.size() == n_consumed) {
                console::set_display(console::reset);
            }

            // if not currently processing queued inputs;
            if ((int) inputTokens.size() <= n_consumed) {

                // check for reverse prompt
                if (inferenceParams.reversePrompts.size()) {
                    std::string last_output;
                    for (auto id : last_n_tokens) {
                        last_output += llama_token_to_str(ctx, id);
                    }

                    is_antiprompt = false;
                    // Check if each of the reverse prompts appears at the end of the output.
                    // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                    // so we'll compensate for that by widening the search window a bit.
                    for (std::string & antiprompt : inferenceParams.reversePrompts) {
                        size_t extra_padding = params.interactive ? 0 : 2;
                        size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                            ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                            : 0;

                        if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos) {
                            if (params.interactive) {
                                is_interacting = true;
                                console::set_display(console::user_input);
                            }
                            is_antiprompt = true;
                            fflush(stdout);
                            break;
                        }
                    }
                }

                // deal with end of text token in interactive mode
                if (last_n_tokens.back() == llama_token_eos()) {
                    if (params.interactive) {
                        if (inferenceParams.reversePrompts.size() != 0) {
                            // tokenize and inject first reverse prompt
                            const auto first_antiprompt = ::llama_tokenize(ctx, inferenceParams.reversePrompts.front(), false);
                            inputTokens.insert(inputTokens.end(), first_antiprompt.begin(), first_antiprompt.end());
                            is_antiprompt = true;
                        }

                        is_interacting = true;
                        printf("\n");
                        console::set_display(console::user_input);
                        fflush(stdout);
                    } else if (params.instruct) {
                        is_interacting = true;
                    }
                }

                if (n_past > 0 && is_interacting) {
                    if (params.instruct) {
                        printf("\n> ");
                    }

                    if (params.input_prefix_bos) {
                        inputTokens.push_back(llama_token_bos());
                    }

                    std::string buffer;
                    if (!params.input_prefix.empty()) {
                        buffer += params.input_prefix;
                        printf("%s", buffer.c_str());
                    }

                    std::string line;
                    bool another_line = true;
                    do {
                        another_line = console::readline(line, params.multiline_input);
                        buffer += line;
                    } while (another_line);

                    // done taking input, reset color
                    console::set_display(console::reset);

                    // Add tokens to tokens only if the input buffer is non-empty
                    // Entering a empty line lets the user pass control back
                    if (buffer.length() > 1) {
                        // append input suffix if any
                        if (!params.input_suffix.empty()) {
                            buffer += params.input_suffix;
                            printf("%s", params.input_suffix.c_str());
                        }

                        // instruct mode: insert instruction prefix
                        if (params.instruct && !is_antiprompt) {
                            n_consumed = inputTokens.size();
                            inputTokens.insert(inputTokens.end(), inp_pfx.begin(), inp_pfx.end());
                        }

                        auto line_inp = ::llama_tokenize(ctx, buffer, false);
                        inputTokens.insert(inputTokens.end(), line_inp.begin(), line_inp.end());

                        // instruct mode: insert response suffix
                        if (params.instruct) {
                            inputTokens.insert(inputTokens.end(), inp_sfx.begin(), inp_sfx.end());
                        }

                        n_remain -= line_inp.size();
                    }

                    input_echo = false; // do not echo this again
                }

                if (n_past > 0) {
                    if (is_interacting) {
                        // reset grammar state if we're restarting generation
                        if (grammar != NULL) {
                            llama_grammar_free(grammar);

                            std::vector<const llama_grammar_element *> grammar_rules(
                                parsed_grammar.c_rules());
                            grammar = llama_grammar_init(
                                grammar_rules.data(), grammar_rules.size(),
                                parsed_grammar.symbol_ids.at("root"));
                        }
                    }
                    is_interacting = false;
                }
            }

            // end of text token
            if (!tokens.empty() && tokens.back() == llama_token_eos() && !(params.instruct || params.interactive)) {
                fprintf(stderr, " [end of text]\n");
                break;
            }

            // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
            if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
                n_remain = params.n_predict;
                is_interacting = true;
            }
            * */

        }

        // save session
        // if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        //        fprintf(stderr, "\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        //        llama_save_session_file(ctx, path_session.c_str(), sessionTokens.data(), sessionTokens.size());
        //    }

        lib.llama_print_timings(ctx)
        if (guidanceCtx != null)
            lib.llama_free(ctx)
        globalCtx.set(null)
        lib.llama_free(ctx)
        lib.llama_free_model(model)
        lib.llama_backend_free()
    }


}
