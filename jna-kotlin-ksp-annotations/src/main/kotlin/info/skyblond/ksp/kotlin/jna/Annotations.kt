package info.skyblond.ksp.kotlin.jna

@Suppress("unused")
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
annotation class JNAStructure(
    val fieldOrder: Array<String> = []
)

@Suppress("unused")
@Target(
    AnnotationTarget.TYPEALIAS, AnnotationTarget.CLASS, AnnotationTarget.PROPERTY, AnnotationTarget.VALUE_PARAMETER,
)
@Retention(AnnotationRetention.SOURCE)
annotation class DefaultValue(val expression: String)