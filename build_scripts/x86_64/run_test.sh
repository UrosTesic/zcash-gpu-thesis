#!/bin/bash

cargo clean --manifest-path="../../librustzcash/Cargo.toml"
cargo build --manifest-path="../../librustzcash/Cargo.toml" --lib --release
cd ../../librustzcash/bellman/
g++ -pthread -Wl,--no-as-needed -ldl -lOpenCL ../../librustzcash/test_multiexp.c -L../../librustzcash/target/release -lrustzcash
./a.out

