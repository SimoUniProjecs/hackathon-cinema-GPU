#!/bin/bash

for n in $(seq 500 100 10000); do
    echo "Eseguendo ./jacobi -x $n -y $n"
    ./jacobi_cuda -x "$n" -y "$n"
done
