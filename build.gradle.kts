plugins {
    kotlin("jvm") version "1.9.0"
    id("com.google.devtools.ksp") version "1.9.0-1.0.12"
}

group = "info.skyblond"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("net.java.dev.jna:jna:5.13.0")
    implementation("net.java.dev.jna:jna-platform:5.13.0")

    implementation(project(":jna-kotlin-ksp-annotations"))
    ksp(project(":jna-kotlin-ksp"))

    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}
