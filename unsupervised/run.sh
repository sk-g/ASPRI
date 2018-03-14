#!/bin/bash
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=50 --window=1 --min_count=2 --size=128
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=80 --window=1 --min_count=2 --size=128
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=100 --window=1 --min_count=2 --size=128
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=1000 --window=1 --min_count=2 --size=128 --save=1 --dump=1
PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=5000 --window=1 --min_count=2 --size=128
#PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=100 --window=1 --min_count=1 --size=64
#PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=16
#PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=32
#PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=32 --true_paths=11012018.txt --fake_paths=24012018_f.txt
#PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=32 --true_paths=11012018.txt --fake_paths=24012018.txt
#PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=32 --true_paths=11012018_f.txt --fake_paths=24012018.txt
#PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=32 --true_paths=11012018_f.txt --fake_paths=24012018_f.txt
