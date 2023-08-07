package info.skyblond.libllama.example

import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference
import info.skyblond.libllama.*
import io.javalin.Javalin
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlin.math.max

/**
 * A very simple example of multi-user chat bot.
 * Note: This is not thread-safe and will cause racing conditions.
 * */
object ChatServer {
    private val lib: LibLLaMa

    init {
        System.setProperty("jna.library.path", "./")
        lib = Native.load("llama", LibLLaMa::class.java) as LibLLaMa
    }

    private val systemPrompt = """
        |Text transcript of a never ending dialog, where USER interacts with an AI assistant named ChatLLaMa.
        |ChatLLaMa is helpful, kind, honest, friendly, good at writing and never fails to answer USER's requests immediately and with details and precision.
        |There are no annotations like (30 seconds passed...) or (to himself), just what USER and ChatLLaMa say aloud to each other.
        |The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.
        |The transcript only includes text, it does not include markup like HTML and Markdown.
    """.trimMargin()

    private val modelParams = ModelParams(
        modelPath = "/data/llama-model/ggml-llama-2-7b-chat-f16.bin"
    )
    private val contextParameter = lib.getContextParams(
        gqa = 1,
        contextSize = 4096,
        batchSize = 512,
        rmsNormEps = 1e-5f,
        nGpuLayers = 10,
    ).also { println("Seed: ${it.seed}") }
    private val inferenceParams = InferenceParams(
        nKeep = 0,
        nPredict = 10000,
        inputPrefixBOS = true,
        inputPrefix = "[INST]",
        inputSuffix = "[/INST]",
    )
    private val samplingParams = SamplingParams(
        temp = 0.7f,
        topK = 40,
        topP = 0.5f,
        repeatPenalty = 1.22f,
        repeatLastN = 256,
    )
    private val nThread = getProcessorCount()

    private val systemSessionFile = File("./server-session/base")

    private fun prepareSystemCtx(ctx: llama_context): List<Int> {
        val prompt = " <<SYS>>\n${systemPrompt}\n<</SYS>>\n\n"
        val promptTokenized = lib.tokenize(ctx, prompt, true).toMutableList()
        val inferenceCore = InferenceCore(contextParameter, inferenceParams, promptTokenized.size)

        val sessionTokens = mutableListOf<Int>()
        val lastNTokens = RingTokenBuffer(lib.llama_n_ctx(ctx))
        lastNTokens.addAll(promptTokenized)
        inferenceCore.inference(lib, lastNTokens, ctx, promptTokenized, sessionTokens, nThread)

        // save the base session
        lib.llama_save_session_file(
            ctx, systemSessionFile.also { it.parentFile.mkdirs() }.path,
            sessionTokens.toIntArray(), sessionTokens.size
        )

        return promptTokenized
    }

    private fun loadUserSession(model: llama_model, name: String): Pair<llama_context, MutableList<Int>> {
        val ctx = lib.llama_new_context_with_model(model, contextParameter)
        val userSessionFile = File("./server-session/user-$name")
        val sessionTokens = (if (userSessionFile.exists()) userSessionFile else systemSessionFile).let {
            val data = IntArray(contextParameter.n_ctx)
            val nTokenCountOut = IntByReference(0)
            check(
                lib.llama_load_session_file(
                    ctx, it.path,
                    data, contextParameter.n_ctx, nTokenCountOut
                ).toInt() == 1
            ) { "failed to load session file '${it.path}'" }
            lib.llama_set_rng_seed(ctx, contextParameter.seed)
            data.copyOf(nTokenCountOut.value).toMutableList()
        }

        return ctx to sessionTokens
    }

    private fun doUserInference(
        ctx: llama_context, sessionTokens: MutableList<Int>, nKeep: Int,
        userInput: String
    ): String {
        val lastNTokens = RingTokenBuffer(lib.llama_n_ctx(ctx))
        val inferenceCore = InferenceCore(contextParameter, inferenceParams, nKeep)
        // the buffer of unprocessed tokens
        val tokens = mutableListOf<Int>()
        tokens.addAll(sessionTokens)
        lastNTokens.addAll(sessionTokens)
        val buffer = StringBuilder()
        if (userInput.isNotEmpty()) {
            // processing input
            if (inferenceParams.inputPrefixBOS) tokens.add(lib.llama_token_bos())
            if (inferenceParams.inputPrefix.isNotEmpty()) {
                buffer.append(inferenceParams.inputPrefix)
            }
            buffer.append(userInput)
            // append input suffix if any
            if (inferenceParams.inputSuffix.isNotEmpty()) {
                buffer.append(inferenceParams.inputSuffix)
            }
            // tokenize input
            val bufferTokenized = lib.tokenize(ctx, buffer.toString(), false)
            tokens.addAll(bufferTokenized.toList())
        }

        val generatedTextBuffer = StringBuilder()
        while (inferenceCore.canInference(false)) {
            // process all the input and predict next token
            inferenceCore.inference(lib, lastNTokens, ctx, tokens, sessionTokens, nThread)
            // now we processed all unprocessed tokens, clear the buffer
            tokens.clear()
            val tokenId = inferenceCore.sample(lib, ctx, samplingParams, lastNTokens)
            // add it to the context
            tokens.add(tokenId)

            // here we rely on eos to stop, you may also search the reverse prompt
            if (tokenId == lib.llama_token_eos()) break

            generatedTextBuffer.append(lib.llama_token_to_str(ctx, tokenId))
        }

        return generatedTextBuffer.toString()
    }

    private fun saveUserSession(
        name: String, ctx: llama_context, sessionTokens: MutableList<Int>
    ) {
        // drop old data to save some disk space
        val maxSize = max(lib.llama_n_ctx(ctx), lib.llama_get_kv_cache_token_count(ctx))
        while (sessionTokens.size > maxSize) sessionTokens.removeFirst()
        val userSessionFile = File("./server-session/user-$name")
            .also { it.parentFile.mkdirs() }
        lib.llama_save_session_file(
            ctx, userSessionFile.path,
            sessionTokens.toIntArray(), sessionTokens.size
        )
        lib.llama_free(ctx)
        System.gc() // suggest a gc
    }

    @JvmStatic
    fun main(args: Array<String>) {
        lib.initLLaMaBackend()
        val (model, systemCtx) = lib.loadModelAndContextWithParams(contextParameter, modelParams, nThread)
        println("Preparing base context...")
        val initialPrompt = prepareSystemCtx(systemCtx)
        lib.llama_free(systemCtx)

        // calculate the actual nKeep based on parameters and current setup
        val nKeep = if (inferenceParams.nKeep !in 0..initialPrompt.size || inferenceParams.alpacaInstruct) {
            initialPrompt.size // in this case, we reset the nKeep to actual input size
        } else inferenceParams.nKeep

        val queue = LinkedBlockingQueue<Triple<String, String, AtomicReference<String>>>()

        thread {
            Javalin.create()
                .post("/inference/{name}") {
                    val name = it.pathParam("name")
                    val userInput = it.body()
                    val result = AtomicReference<String>()
                    queue.offer(Triple(name, userInput, result))
                    while (result.get() == null)
                        Thread.sleep(500)
                    it.result(result.get())
                }
                .start("0.0.0.0", 7070)
        }

        // must handle on main thread,
        // since all C resources are bind to main thread
        // (if we have more resources, we can initial the llamacpp on multiple threads)
        // This will be super slow, maybe use websocket will give a better responsiveness
        while (true) {
            val (name, userInput, result) = queue.take()
            // due to the limited resource, make sure each time only
            // one request is processed
            println("Loading user context: $name")
            val (ctx, session) = loadUserSession(model, name)
            println("Doing user inference: $name, $userInput")
            result.set(doUserInference(ctx, session, nKeep, userInput))
            saveUserSession(name, ctx, session)
            println("Finished user inference: $name")
        }

        println("Main done!")

        lib.llama_free_model(model)
        lib.llama_backend_free()
    }
}
