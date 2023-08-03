package info.skyblond.ksp.kotlin.jna

import com.google.devtools.ksp.processing.SymbolProcessor
import com.google.devtools.ksp.processing.SymbolProcessorEnvironment
import com.google.devtools.ksp.processing.SymbolProcessorProvider

class JNAKotlinProcessorProvider : SymbolProcessorProvider {
    override fun create(environment: SymbolProcessorEnvironment): SymbolProcessor =
        JNAKotlinProcessor(
            codeGenerator = environment.codeGenerator,
            logger = environment.logger,
//            options = environment.options
        )
}