#!/bin/bash

for n in $(seq 10 1 200); do
    echo "Eseguendo ./jacobi -x $n -y $n"
    ./jacobi -x "$n" -y "$n"
done
