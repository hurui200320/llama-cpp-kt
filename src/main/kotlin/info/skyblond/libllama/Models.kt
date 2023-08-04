@file:Suppress("SpellCheckingInspection", "ClassName", "ArrayInDataClass", "unused")

package info.skyblond.libllama

import com.sun.jna.Callback
import com.sun.jna.Pointer
import info.skyblond.ksp.kotlin.jna.DefaultValue
import info.skyblond.ksp.kotlin.jna.JNAStructure


// Note: All 'bool' in C++ must be "Byte" in Java.

@DefaultValue("Pointer.NULL")
typealias llama_model = Pointer?
@DefaultValue("Pointer.NULL")
typealias llama_context = Pointer?

/**
 * [id]: token id.
 *
 * [logit]: log-odds of the token.
 *
 * [p]: probability of the token
 * */
@JNAStructure(["id", "logit", "p"])
data class JNAllama_token_data(
    val id: Int,
    val logit: Float,
    val p: Float
)

/**
 * [data] is array of [llama_token_data].
 * */
@JNAStructure(["data", "size", "sorted"])
data class JNAllama_token_data_array(
    val data: Pointer,
    val size: Int,
    val sorted: Byte
)

@DefaultValue("#fqName { _,_ -> }")
fun interface llama_progress_callback : Callback {
    fun invoke(progress: Float, ctx: Pointer)
}

@JNAStructure
data class JNAllama_context_params(
    val seed: Int,
    val n_ctx: Int,
    val n_batch: Int,
    val n_gqa: Int,
    val rms_norm_eps: Float,
    val n_gpu_layers: Int,
    val main_gpu: Int,

    @DefaultValue("FloatArray(info.skyblond.libllama.LibLLaMa.LLAMA_MAX_DEVICES)")
    val tensor_split: FloatArray,

    val rope_freq_base: Float,
    val rope_freq_scale: Float,

    val progress_callback: llama_progress_callback,
    val progress_callback_user_data: Pointer?,

    val low_vram: Byte,
    val mul_mat_q: Byte,
    val f16_kv: Byte,
    val logits_all: Byte,
    val vocab_only: Byte,
    val use_mmap: Byte,
    val use_mlock: Byte,
    val embedding: Byte,
)

enum class llama_ftype(val n: Int) {
    LLAMA_FTYPE_ALL_F32(0),
    LLAMA_FTYPE_MOSTLY_F16(1), // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0(2), // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1(3), // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16(4), // tok_embeddings.weight and output.weight are F16

    // LLAMA_FTYPE_MOSTLY_Q4_2(5), // support has been removed
    // LLAMA_FTYPE_MOSTLY_Q4_3(6), // support has been removed

    LLAMA_FTYPE_MOSTLY_Q8_0(7), // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_0(8), // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_1(9), // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K(10),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_S(11),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_M(12),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_L(13),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_S(14),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_M(15),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_S(16),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_M(17),// except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q6_K(18),// except 1d tensors
}

fun Int.llama_ftype(): llama_ftype? = llama_ftype.entries.find { it.n == this }

/**
 * [ftype] is [llama_ftype]
 * */
@JNAStructure(["nthread", "ftype", "allow_requantize", "quantize_output_tensor"])
data class JNAllama_model_quantize_params(
    val nthread: Int,
    val ftype: Int,
    val allow_requantize: Byte,
    val quantize_output_tensor: Byte
)

@DefaultValue("Pointer.NULL")
typealias llama_grammar = Pointer

enum class llama_gretype(val n: Int) {
    // end of rule definition
    LLAMA_GRETYPE_END(0),

    // start of alternate definition for rule
    LLAMA_GRETYPE_ALT(1),

    // non-terminal element: reference to rule
    LLAMA_GRETYPE_RULE_REF(2),

    // terminal element: character (code point)
    LLAMA_GRETYPE_CHAR(3),

    // inverse char(s) ([^a], [^a-b] [^abc])
    LLAMA_GRETYPE_CHAR_NOT(4),

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    LLAMA_GRETYPE_CHAR_RNG_UPPER(5),

    // modifies a preceding LLAMA_GRETYPE_CHAR or
    // LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    LLAMA_GRETYPE_CHAR_ALT(6),
}

fun Int.llama_gretype(): llama_gretype? = llama_gretype.entries.find { it.n == this }

@JNAStructure(["type", "value"])
data class JNAllama_grammar_element(
    // llama_gretype
    val type: Int,
    // Unicode code point or rule ID
    val value: Int
)

@JNAStructure
data class JNAllama_timings(
    val t_start_ms: Double,
    val t_end_ms: Double,
    val t_load_ms: Double,
    val t_sample_ms: Double,
    val t_p_eval_ms: Double,
    val t_eval_ms: Double,

    val n_sample: Int,
    val n_p_eval: Int,
    val n_eval: Int,
)
