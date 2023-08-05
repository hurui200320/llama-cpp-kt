package info.skyblond.libllama.example

import com.sun.jna.Native
import info.skyblond.libllama.*
import java.io.File

/**
 * The quantize.cpp
 * */
fun LibLLaMa.quantize(
    model: File, ftype: llama_ftype, output: File,
    leaveOutputTensor: Boolean = false,
    allowRequantize: Boolean = false,
    nThread: Int = getProcessorCount()
) {
    val params = llama_model_quantize_default_params()
    if (leaveOutputTensor) params.quantize_output_tensor = 0
    if (allowRequantize) params.allow_requantize = 1

    initLLaMaBackend() // assuming no numa system

    params.nthread = nThread
    params.ftype = ftype.n

    val mainStartTime = llama_time_us()
    var quantizeDuration = 0L

    // load model
    run {
        val startTime = llama_time_us()
        val res = llama_model_quantize(
            model.absolutePath, output.absolutePath,
            llama_model_quantize_params.ByReference(params.pointer)
        )
        if (res != 0) error("Failed to quantize model")

        quantizeDuration = llama_time_us() - startTime
    }

    // report timing
    run {
        val mainEndTime = llama_time_us()

        println()
        println("quantize time = %8.2f ms".format(quantizeDuration / 1000.0))
        println("total time = %8.2f ms".format(mainEndTime - mainStartTime / 1000.0))
    }

    llama_backend_free()
}

fun main() {
    System.setProperty("jna.library.path", "./")
    val lib = Native.load("llama", LibLLaMa::class.java) as LibLLaMa

    val inputModel = File("/data/llama-model/ggml-llama-2-7b-chat-f16.bin")
    val ftype = llama_ftype.LLAMA_FTYPE_MOSTLY_Q8_0
    val output = File("/data/llama-model/ggml-llama-2-7b-chat-q8_0.bin")

    lib.quantize(inputModel, ftype, output)
}