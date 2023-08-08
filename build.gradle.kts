plugins {
    kotlin("jvm") version "1.9.0"
    id("com.google.devtools.ksp") version "1.9.0-1.0.12"
    `maven-publish`
}

allprojects {
    group = "info.skyblond"
    version = "0.0.1"
}

repositories {
    mavenCentral()
}

dependencies {
    api("net.java.dev.jna:jna:5.13.0")

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

publishing {
    publications {
        create<MavenPublication>("llama-cpp-kt") {
            groupId = project.group as String
            artifactId = project.name
            version = project.version as String
            from(components["java"])
        }
    }
}
