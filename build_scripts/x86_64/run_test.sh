#!/bin/bash

cargo clean --manifest-path="/home/utesic/Projects/zcash/librustzcash/Cargo.toml"
cargo build --manifest-path="/home/utesic/Projects/zcash/librustzcash/Cargo.toml" --lib --release
cd /home/utesic/Projects/zcash/librustzcash/bellman/
g++ -pthread -Wl,--no-as-needed -ldl -lOpenCL /home/utesic/Projects/zcash/librustzcash/test_multiexp.c -L/home/utesic/Projects/zcash/librustzcash/target/release -lrustzcash
./a.out

