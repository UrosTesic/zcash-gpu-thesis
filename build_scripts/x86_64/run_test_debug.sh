#!/bin/bash

cargo build --manifest-path="/home/utesic/Projects/zcash/librustzcash/Cargo.toml" --lib
cd /home/utesic/Projects/zcash/librustzcash/bellman/
g++ -g -pthread -Wl,--no-as-needed -ldl -lOpenCL /home/utesic/Projects/zcash/librustzcash/test_multiexp.c -L/home/utesic/Projects/zcash/librustzcash/target/debug -lrustzcash
./a.out

