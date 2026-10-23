#!/bin/bash
# Compile the two C/C++ shared libraries used by the Python code.
set -e
cd "$(dirname "$0")/03_Code/src"
g++ -O3 -march=native -fPIC -shared -o cec17/libcec17.so cec17/cec17_wrapper.cpp
gcc -O3 -march=native -fPIC -shared -o tsp/libtsp.so tsp/tsp_meta.c -lm
python3 ../scripts/verify_cec17.py
