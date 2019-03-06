#!/bin/bash
LIBRUSTZCASH_CARGO_MANIFEST_PATH="../../librustzcash/librustzcash/Cargo.toml"
BELLMAN_CARGO_MANIFEST_PATH="../../librustzcash/bellman/Cargo.toml"

cargo clean --manifest-path="$BELLMAN_CARGO_MANIFEST_PATH" --verbose
cargo clean --manifest-path="$LIBRUSTZCASH_CARGO_MANIFEST_PATH" --verbose
RUSTFLAGS="-C target-cpu=native" cargo +nightly build --manifest-path="$BELLMAN_CARGO_MANIFEST_PATH"  --lib --release --features="u128-support" --verbose
RUSTFLAGS="-C target-cpu=native" cargo +nightly build --manifest-path="$LIBRUSTZCASH_CARGO_MANIFEST_PATH"  --lib --release --features="u128-support" --verbose
cd ../../librustzcash/bellman/
g++ -march=native -pthread -Wl,--no-as-needed -ldl -lOpenCL ../../librustzcash/test_multiexp.c -L../../librustzcash/target/release -lrustzcash
./a.out

