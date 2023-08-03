package info.skyblond.ksp.kotlin.jna

@Suppress("unused")
@Target(AnnotationTarget.CLASS)
annotation class JNAStructure(
    val fieldOrder: Array<String> = []
)

@Suppress("unused")
@Target(
    AnnotationTarget.TYPEALIAS, AnnotationTarget.CLASS,
)
annotation class DefaultValue(val expression: String)