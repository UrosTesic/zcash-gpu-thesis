#!/bin/bash

export PATH=$PATH:/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin
LIBRUSTZCASH_CARGO_MANIFEST_PATH="../../librustzcash/librustzcash/Cargo.toml"
BELLMAN_CARGO_MANIFEST_PATH="../../librustzcash/bellman/Cargo.toml"
CC="/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-clang++"
WRAPPER_CPP_FILE="../../librustzcash/test_multiexp.c"
LIBRUSTZCASH_PATH="../../librustzcash/target/aarch64-linux-android/debug"
LIBOPENCL_PATH="./"
LIBCPP_PATH="/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/lib64"
OUTPUT_FILE="./aarch64-test-debug"

cargo build --manifest-path="$BELLMAN_CARGO_MANIFEST_PATH" --target=aarch64-linux-android --lib --features="opencl_vendor_mesa" --verbose
cargo build --manifest-path="$LIBRUSTZCASH_CARGO_MANIFEST_PATH" --target=aarch64-linux-android --lib --features="opencl_vendor_mesa" --verbose

$CC -g $WRAPPER_CPP_FILE -L$LIBRUSTZCASH_PATH -lrustzcash -L$LIBOPENCL_PATH -lOpenCL -o $OUTPUT_FILE

