package info.skyblond.ksp.kotlin.jna

import com.google.devtools.ksp.getConstructors
import com.google.devtools.ksp.isPublic
import com.google.devtools.ksp.processing.CodeGenerator
import com.google.devtools.ksp.processing.Dependencies
import com.google.devtools.ksp.processing.KSPLogger
import com.google.devtools.ksp.processing.Resolver
import com.google.devtools.ksp.symbol.KSClassDeclaration
import com.google.devtools.ksp.symbol.KSVisitorVoid

class JNAKotlinVisitor(
    private val codeGenerator: CodeGenerator,
    private val logger: KSPLogger,
    private val resolver: Resolver
) : KSVisitorVoid() {

    override fun visitClassDeclaration(classDeclaration: KSClassDeclaration, data: Unit) {
        val originalClassName = classDeclaration.simpleName.getShortName()
        // the class name must start with "JNA" to avoid namespace conflict
        if (!originalClassName.startsWith("JNA")) return
        val className = originalClassName.removePrefix("JNA")

        // no @JNAStructure annotation
        val fieldOrderList = classDeclaration.getFieldOrderList() ?: return
        // no constructor -> no fields
        if (classDeclaration.primaryConstructor == null) return

        val packageName = classDeclaration.packageName.asString()

        // if file already exists -> we have already generated -> skip
        if (resolver.getNewFiles().any {
                it.fileName == className && it.packageName.asString() == packageName
            }) return

        // now get the properties list in a structure
        val propertiesList = classDeclaration.primaryConstructor!!.parameters.let { parameters ->
            if (fieldOrderList.isEmpty()) {
                // no order, use the constructor parameter list
                parameters.toTypedArray()
            } else {
                Array(fieldOrderList.size) { i ->
                    val fieldName = fieldOrderList[i]
                    // here as a side effect, `it.name` is not null
                    parameters.find { it.name?.asString() == fieldName } ?: error(
                        "Property $fieldName not found in class ${className}, package $packageName"
                    )
                }
            }
        }

        // if we met unknown type (can't resolve) -> might be newly generated -> wait for next round
        if (propertiesList.any { p ->
                val resolved = p.type.resolve()
                val typeUnclear = resolved.declaration.let {
                    it.qualifiedName == null && it.simpleName.asString() == "<Error>"
                }
                val typeParameterUnclear = resolved.arguments.any { ksTypeArgument ->
                    if (ksTypeArgument.type == null) {
                        logger.warn("Null type reference for type $resolved")
                        false
                    } else {
                        ksTypeArgument.type!!.resolve().declaration.let {
                            it.qualifiedName == null && it.simpleName.asString() == "<Error>"
                        }
                    }
                }
                typeUnclear || typeParameterUnclear
            }) return

        // we're clear to generate code

        val outputStream = codeGenerator.createNewFile(
            // Make sure to associate the generated file with sources to keep/maintain it across incremental builds.
            // Learn more about incremental processing in KSP from the official docs:
            // https://kotlinlang.org/docs/ksp-incremental.html
            dependencies = Dependencies(false, *resolver.getAllFiles().toList().toTypedArray()),
            packageName = packageName,
            fileName = className
        )

        outputStream += "package ${packageName}\n"
        outputStream += "import com.sun.jna.Pointer\n"
        outputStream += "import com.sun.jna.Structure\n"
        outputStream += "import com.sun.jna.Structure.FieldOrder\n"
        outputStream += "import com.sun.jna.Structure.ByValue\n"
        outputStream += "import com.sun.jna.Structure.ByReference\n"


        val generatedAnnotationValue = propertiesList.joinToString(", ") { "\"$it\"" }

        outputStream += "@FieldOrder(${generatedAnnotationValue})\n"

        classDeclaration.docString?.let {
            outputStream += "/**${it}*/\n"
        }

        outputStream += "open class $className : Structure {\n"
        outputStream += "    constructor()\n"
        outputStream += "    constructor(p: Pointer) : super(p) { read() }\n"
        outputStream += "    class ByValue : ${className}, Structure.ByValue {\n"
        outputStream += "        constructor()\n"
        outputStream += "        constructor(p: Pointer) : super(p) { read() }\n"
        outputStream += "    }\n"
        outputStream += "    class ByReference : ${className}, Structure.ByReference {\n"
        outputStream += "        constructor()\n"
        outputStream += "        constructor(p: Pointer) : super(p) { read() }\n"
        outputStream += "    }\n"
        outputStream += "\n"

        propertiesList.forEach { parameter ->
            val propertyName = parameter.name!!.asString()
            outputStream += "    @JvmField\n"
            val resolvedType = parameter.type.resolve().declaration
            val typeFQName = resolvedType.qualifiedName!!.asString()

            val nullable = when (typeFQName) {
                "com.sun.jna.Pointer" -> "?"
                else -> ""
            }

            val types = when (typeFQName) {
                "kotlin.Array" -> {
                    val type =
                        parameter.type.resolve().arguments[0].type!!.resolve().declaration.qualifiedName!!.asString()
                    "<${type}?>"
                }

                else -> ""
            }


            val defaultValue = parameter.getDefaultValueExpression() ?: when (typeFQName) {
                "kotlin.Int", "kotlin.Byte" -> "0"
                "kotlin.Boolean" -> "false"
                "kotlin.Float" -> "0.0f"
                "kotlin.Double" -> "0.0"
                "kotlin.FloatArray" -> "floatArrayOf(0.0f)"
                "kotlin.Array" -> "arrayOf(null)"
                "com.sun.jna.Pointer" -> "com.sun.jna.Pointer.NULL"
                else -> {
                    // if it has public no arg constructor, use it
                    val hasNoArgCons = resolver.getClassDeclarationByName(resolvedType.qualifiedName!!)
                        ?.getConstructors()?.any { it.parameters.isEmpty() && it.isPublic() } ?: false
                    if (hasNoArgCons) {
                        "$typeFQName()"
                    } else { // otherwise find the annotation
                        val rawExp = resolvedType.getDefaultValueExpression() ?: error(
                            "Unknown default value for: ${typeFQName}, " +
                                    "property ${propertyName}, class ${className}, package $packageName"
                        )
                        rawExp.replace("#fqName", typeFQName)
                    }
                }
            }

            outputStream += "    final var ${propertyName}: ${typeFQName}${types}${nullable} = ${defaultValue}\n"
            outputStream += "\n"
        }
        outputStream += "}\n"

        outputStream.close()
    }


}

