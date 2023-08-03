package info.skyblond.ksp.kotlin.jna

import com.google.devtools.ksp.symbol.KSClassDeclaration
import com.google.devtools.ksp.symbol.KSDeclaration
import info.skyblond.ksp.kotlin.jna.Constants.DEFAULT_VALUE_ANNOTATION_FQ_NAME
import info.skyblond.ksp.kotlin.jna.Constants.DEFAULT_VALUE_ANNOTATION_SHORT_NAME
import info.skyblond.ksp.kotlin.jna.Constants.DEFAULT_VALUE_ANNOTATION_VALUE_NAME
import info.skyblond.ksp.kotlin.jna.Constants.JNA_ANNOTATION_FQ_NAME
import info.skyblond.ksp.kotlin.jna.Constants.JNA_ANNOTATION_SHORT_NAME
import java.io.OutputStream


operator fun OutputStream.plusAssign(str: String) {
    this.write(str.toByteArray())
}

/**
 * Return null: no annotation.
 * Return empty list: no order defined.
 * */
fun KSClassDeclaration.getFieldOrderList(): List<String>? {
    val annotation = this.annotations.find {
        it.shortName.asString() == JNA_ANNOTATION_SHORT_NAME
                && it.annotationType.resolve().declaration.qualifiedName?.asString() == JNA_ANNOTATION_FQ_NAME
    } ?: return null

    val annotationValueList = annotation.arguments.find {
        it.name?.asString() == Constants.JNA_ANNOTATION_VALUE_NAME
    }?.value as? List<*> ?: return emptyList()

    return annotationValueList.map { it as String }
}


fun KSDeclaration.getDefaultValueExpression(): String? =
    this.annotations.find {
        it.shortName.asString() == DEFAULT_VALUE_ANNOTATION_SHORT_NAME
                && it.annotationType.resolve().declaration.qualifiedName?.asString() == DEFAULT_VALUE_ANNOTATION_FQ_NAME
    }?.arguments?.find { it.name?.asString() == DEFAULT_VALUE_ANNOTATION_VALUE_NAME }?.value as? String

