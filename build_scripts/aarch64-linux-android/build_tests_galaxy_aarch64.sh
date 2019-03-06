#!/bin/bash

# add NDK to PATH
export PATH=$PATH:/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin
LIBRUSTZCASH_CARGO_MANIFEST_PATH="../../librustzcash/librustzcash/Cargo.toml"
BELLMAN_CARGO_MANIFEST_PATH="../../librustzcash/bellman/Cargo.toml"
CC="/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-clang++"
WRAPPER_CPP_FILE="../../librustzcash/test_multiexp.c"
LIBRUSTZCASH_PATH="../../librustzcash/target/aarch64-linux-android/release"
LIBOPENCL_PATH="./"
LIBCPP_PATH="/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/lib64"
OUTPUT_FILE="./aarch64-test-release"

cargo build --manifest-path="$BELLMAN_CARGO_MANIFEST_PATH" --target=aarch64-linux-android --lib --release --features="opencl_vendor_mesa" --verbose
cargo build --manifest-path="$LIBRUSTZCASH_CARGO_MANIFEST_PATH" --target=aarch64-linux-android --lib --release --features="opencl_vendor_mesa" --verbose

$CC -O3 $WRAPPER_CPP_FILE -I/home/utesic/Android/Sdk/ndk-bundle/sources/cxx-stl/llvm-libc++/include/ -stdlib=libstdc++ -L$LIBRUSTZCASH_PATH -lrustzcash -L$LIBOPENCL_PATH -lOpenCL -fuse-ld=gold -o $OUTPUT_FILE
#-lc++ -lcutils -lnativewindow -llog -lz -lion_exynos -lutils -lui -lhardware -Wl,-rpath,./
