#!/bin/bash

cargo build --manifest-path="../../librustzcash/Cargo.toml" --lib
cd ../../librustzcash/bellman/
g++ -g -pthread -Wl,--no-as-needed -ldl -lOpenCL ../../librustzcash/test_multiexp.c -L../../librustzcash/target/debug -lrustzcash
./a.out

