#!/bin/bash
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=8
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=100 --window=1 --min_count=1 --size=64
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=16
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=32
