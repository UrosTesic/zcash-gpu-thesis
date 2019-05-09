#!/bin/bash

cargo clean --target=i686-unknown-linux-gnu --manifest-path="../../librustzcash/Cargo.toml"
cargo build --target=i686-unknown-linux-gnu --manifest-path="../../librustzcash/Cargo.toml" --lib --release
cd ../../librustzcash/bellman/
g++ -m32 -pthread -Wl,--no-as-needed -ldl -L/home/utesic/Projects/zcash/thesis/zcash-gpu-thesis/build_scripts/x86 -lOpenCL ../../librustzcash/test_multiexp.c -L../../librustzcash/target/i686-unknown-linux-gnu/release -lrustzcash
LD_LIBRARY_PATH=/home/utesic/Projects/zcash/thesis/zcash-gpu-thesis/build_scripts/x86 ./a.out

