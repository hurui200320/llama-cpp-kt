package info.skyblond.ksp.kotlin.jna

import com.google.devtools.ksp.processing.CodeGenerator
import com.google.devtools.ksp.processing.KSPLogger
import com.google.devtools.ksp.processing.Resolver
import com.google.devtools.ksp.processing.SymbolProcessor
import com.google.devtools.ksp.symbol.KSAnnotated
import com.google.devtools.ksp.validate

class JNAKotlinProcessor(
    private val codeGenerator: CodeGenerator,
    private val logger: KSPLogger,
//    private val options: Map<String, String>
) : SymbolProcessor {

    override fun process(resolver: Resolver): List<KSAnnotated> {
        val symbols = resolver
            .getSymbolsWithAnnotation(Constants.JNA_ANNOTATION_FQ_NAME)

        symbols.forEach {
            it.accept(JNAKotlinVisitor(codeGenerator, logger, resolver), Unit)
        }

        Constants.outputStreamPool.forEach { (_, o) -> o.close() }
        return symbols.filterNot { it.validate() }.toList()
    }
}