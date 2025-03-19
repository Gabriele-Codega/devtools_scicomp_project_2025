#!/bin/bash

rank=$OMPI_COMM_WORLD_RANK

export NUMBA_NUM_THREADS=1

kernprof -lz -o "out_big_$rank.lprof" scripts/run.py --config experiments/config
# valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file="logs/memory/512_block_rank_$rank.log" python scripts/run.py --config experiments/config
