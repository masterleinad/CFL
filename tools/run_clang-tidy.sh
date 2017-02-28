#!/bin/bash
find .. -name '*.cc' -exec \
  clang-tidy -p ./ {} -header-filter=../../* \
             -checks=* -- -DDEBUG -I/opt/compiler/gcc-6.2.0/include/c++/6.2.0 \
    -I/opt/compiler/gcc-6.2.0/include/c++/6.2.0/x86_64-pc-linux-gnu/ \
    -L/opt/compiler/gcc-6.2.0/lib -I../.. -I/mnt/data/darndt/deal.II-clang-4/include \
    -I/mnt/data/darndt/deal.II-clang-4/include/deal.II/bundled \
    -I/mnt/data/darndt/Sources/openmpi-clang-4.0.0/include \
    -I/mnt/data/darndt/Trilinos-12.6.1-clang-4.0.0/include \
    -I/usr/include/suitesparse -I/mnt/data/darndt/petsc-3.6.3-clang-4.0.0/include \
    -I/mnt/data/darndt/petsc-3.6.3-clang-4.0.0/arch-linux2-c-opt/include \
    -I/mnt/data/darndt/p4est-1.1-clang-4.0.0/FAST/include  -Wfatal-errors \
    -pedantic -fPIC -Wall -Wextra -Wpointer-arith -Wwrite-strings -Wsynth \
    -Wsign-compare -Wswitch -Woverloaded-virtual -Wno-long-long \
    -Qunused-arguments -Wno-unsupported-friend -Wno-unused-parameter \
    -Wno-unused-variable -Wno-c99-extensions -Wno-variadic-macros -Wno-c++11-extensions \
    -Wno-undefined-var-template -Wno-unused-function -openmp-simd -std=c++14 -std=c++1z \
    -march=native -Wno-unused-local-typedefs -std=c++1z -Og -ggdb -Wa,--compress-debug-sections \;
