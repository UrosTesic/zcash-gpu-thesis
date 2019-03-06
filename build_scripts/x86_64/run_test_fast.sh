#!/bin/bash
LIBRUSTZCASH_CARGO_MANIFEST_PATH="/home/utesic/Projects/zcash/librustzcash/librustzcash/Cargo.toml"
BELLMAN_CARGO_MANIFEST_PATH="/home/utesic/Projects/zcash/librustzcash/bellman/Cargo.toml"

cargo clean --manifest-path="$BELLMAN_CARGO_MANIFEST_PATH" --verbose
cargo clean --manifest-path="$LIBRUSTZCASH_CARGO_MANIFEST_PATH" --verbose
RUSTFLAGS="-C target-cpu=native" cargo +nightly build --manifest-path="$BELLMAN_CARGO_MANIFEST_PATH"  --lib --release --features="u128-support" --verbose
RUSTFLAGS="-C target-cpu=native" cargo +nightly build --manifest-path="$LIBRUSTZCASH_CARGO_MANIFEST_PATH"  --lib --release --features="u128-support" --verbose
cd /home/utesic/Projects/zcash/librustzcash/bellman/
g++ -march=native -pthread -Wl,--no-as-needed -ldl -lOpenCL /home/utesic/Projects/zcash/librustzcash/test_multiexp.c -L/home/utesic/Projects/zcash/librustzcash/target/release -lrustzcash
./a.out

