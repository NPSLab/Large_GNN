#!/bin/bash
for file in configs/main_exps/sage/arxiv/final_configs/*
do
    rm -r datasets/ogbn_arxiv/processed/
    echo $file 2>&1 | tee -a final_arxiv_sage.log
    python run_ogbn.py with $file 2>&1 | tee -a final_arxiv_sage.log
done
