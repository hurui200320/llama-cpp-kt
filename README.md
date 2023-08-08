# llama-cpp-kt

[![](https://jitpack.io/v/info.skyblond/llama-cpp-kt.svg)](https://jitpack.io/#info.skyblond/llama-cpp-kt)

The Kotlin wrapper of [llama.cpp](https://github.com/ggerganov/llama.cpp), powered by JNA.

## Setup

First, you need to build your own `libllama.so` from [llama.cpp](https://github.com/ggerganov/llama.cpp), using cmake:

```bash
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON <other flags>
cmake --build . --config Release
```

You need the `-DBUILD_SHARED_LIBS=ON` to build shared lib (`.so`), otherwise it will build the static library (`.a`),
which cannot be loaded by JNA.

Then see [jitpack](https://jitpack.io/#info.skyblond/llama-cpp-kt/) document for how to use the build artifacts in your maven or gradle project.

## Usage

First to set the JNA library path:

```kotlin
init {
    System.setProperty("jna.library.path", "./")
    lib = LibLLaMa.LIB
}
```

This will load lib `llama` by default, aka the JNA will search for `libllama.so` or `llama.dll`. If you have a different
file name, you may use `Native.load("llama", LibLLaMa::class.java) as LibLLaMa` to get your own instance. But do notice that the code requires the default instance to work, since some constant are decided at runtime (for example the `LLAMA_MAX_DEVICES` is 1 when using CPU but will be 16 when using cuda).

This is a low level binding, which means you get the full control of the C functions, which looks like this:

+ `lib.llama_model_quantize_default_params()`
+ `lib.llama_load_session_file(ctx, sessionPath, tokens, size, pInt)`

There are also some high level helper functions like:

+ `lib.initLLaMaBackend()`
+ `lib.getContextParams(contextSize=1024, rmsNormEps=1e-5f)`

You can check the [example](https://github.com/hurui200320/llama-cpp-kt/tree/master/examples) subproject to see how to use it. I implemented the original [`quantize.cpp`](https://github.com/hurui200320/llama-cpp-kt/blob/master/examples/src/main/kotlin/info/skyblond/libllama/example/Quantize.kt), [`simple.cpp`](https://github.com/hurui200320/llama-cpp-kt/blob/master/examples/src/main/kotlin/info/skyblond/libllama/example/Simple.kt)
and the [`main.cpp`](https://github.com/hurui200320/llama-cpp-kt/blob/master/examples/src/main/kotlin/info/skyblond/libllama/example/Main.kt). There are also [a simple multi-session chat server](https://github.com/hurui200320/llama-cpp-kt/blob/master/examples/src/main/kotlin/info/skyblond/libllama/example/ChatServer.kt) which can serve multiple session at once over HTTP (in the example I use single thread computation since I don't have enough ram to do parallel computing).

## Roadmap

Currently this repo is still very new and I don't have that many ideas on which path to go. So discussions and
contributions are welcome. JVM is a fantastic platform but apparently it is underestimated during the machine learning rush. Despite Python and C++ can create powerful and fast computing deep learning models, I still believe JVM is the best platform to develop complex business logic. Here I choose Kotlin because it's much better than Java yet maintained a good interoperability with Java (however I don't think you can use this lib in Java).

One clear objective is get rid of JNA when Foreign Function & Memory API is in stable release (Maybe JDK 21?).

Another objective is make the grammar working. Currently the grammar feature is missing (Now you can use grammar related call, but the parser is not there. So you have to grow your own tree).
