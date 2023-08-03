package info.skyblond.ksp.kotlin.jna

import java.io.OutputStream
import java.util.concurrent.ConcurrentHashMap

object Constants {
    const val JNA_ANNOTATION_VALUE_NAME = "fieldOrder"
    val JNA_ANNOTATION_KLASS = JNAStructure::class
    val JNA_ANNOTATION_FQ_NAME = JNA_ANNOTATION_KLASS.qualifiedName!!
    val JNA_ANNOTATION_SHORT_NAME = JNA_ANNOTATION_KLASS.simpleName!!

    const val DEFAULT_VALUE_ANNOTATION_VALUE_NAME = "expression"
    val DEFAULT_VALUE_ANNOTATION_KLASS = DefaultValue::class
    val DEFAULT_VALUE_ANNOTATION_FQ_NAME = DEFAULT_VALUE_ANNOTATION_KLASS.qualifiedName!!
    val DEFAULT_VALUE_ANNOTATION_SHORT_NAME = DEFAULT_VALUE_ANNOTATION_KLASS.simpleName!!

    val outputStreamPool = ConcurrentHashMap<String, OutputStream>()
}