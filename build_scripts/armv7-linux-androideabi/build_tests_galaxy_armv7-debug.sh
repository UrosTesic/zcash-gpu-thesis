#!/bin/bash

export PATH=$PATH:/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin
LIBRUSTZCASH_CARGO_MANIFEST_PATH="/home/utesic/Projects/zcash/librustzcash/librustzcash/Cargo.toml"
BELLMAN_CARGO_MANIFEST_PATH="/home/utesic/Projects/zcash/librustzcash/bellman/Cargo.toml"
CC="/home/utesic/Android/Sdk/ndk-bundle/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi16-clang++"
WRAPPER_CPP_FILE="/home/utesic/Projects/zcash/librustzcash/test_multiexp.c"
LIBRUSTZCASH_PATH="/home/utesic/Projects/zcash/librustzcash/target/armv7-linux-androideabi/debug"
LIBOPENCL_PATH="/home/utesic/Projects/zcash/build_scripts/armv7-linux-androideabi"
OUTPUT_FILE="/home/utesic/Projects/zcash/build_scripts/armv7-linux-androideabi/armv7-test-debug"

cargo build --manifest-path="$BELLMAN_CARGO_MANIFEST_PATH" --target=armv7-linux-androideabi --lib --features="opencl_vendor_mesa" --verbose
cargo build --manifest-path="$LIBRUSTZCASH_CARGO_MANIFEST_PATH" --target=armv7-linux-androideabi --lib --features="opencl_vendor_mesa" --verbose

$CC -g $WRAPPER_CPP_FILE -L$LIBRUSTZCASH_PATH -lrustzcash -L$LIBOPENCL_PATH -lOpenCL -o $OUTPUT_FILE

