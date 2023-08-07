plugins {
    kotlin("jvm")
    `maven-publish`
}

group = "info.skyblond"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("net.java.dev.jna:jna:5.13.0")
    implementation("com.google.devtools.ksp:symbol-processing-api:1.9.0-1.0.12")

    implementation(project(":jna-kotlin-ksp-annotations"))

    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(8)
}

tasks.test {
    useJUnitPlatform()
}