pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.5.0"
}

rootProject.name = "llama-cpp-kt"
include("jna-kotlin-ksp")
include("jna-kotlin-ksp-annotations")
include("examples")
