from multiprocess import Pool, Manager, managers
import numpy as np
import psutil
import sys
from rich import print as pprint
import pandas as pd
from tqdm import tqdm
import sympy as sp
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import *
from copy import deepcopy
from time import perf_counter
import json
import warnings
warnings.filterwarnings("ignore")

from anuta.constructor import Constructor, Millisampler, Cidds001, Netflix
from anuta.theory import Theory
from anuta.miner import miner_versionspace, miner_valiant, validator

    
def main(constructor: Constructor, refconstructor: Constructor, limit: int):
    # miner_valiant(constructor, limit)
    miner_versionspace(constructor, refconstructor, limit)

if __name__ == '__main__':
    # boundsfile = f"./data/meta_bounds.json"
    # filepath = f"./data/meta_w10_s5_{sys.argv[1]}.csv"
    # millisampler = Millisampler(filepath)
    # main(millisampler, sys.argv[1])
    
    if len(sys.argv) == 4 and sys.argv[1] == 'validate':
        n = 1000
        rulepath = sys.argv[2]
        assert rulepath.endswith('.rule'), "Invalid rule file."
        rules = Theory.load_constraints(rulepath, False)
        rule_label = int(rulepath.split('_')[-2])
        
        validation_data = sys.argv[3]
        assert validation_data.endswith('.csv'), "Invalid validation data."
        # cidds = Cidds001(validation_data)
        netflix = Netflix(validation_data)
        data_label = validation_data.split('/')[-1].split('.')[-2]
        if 'syn' in validation_data:
            n = None
        netflix.df = netflix.df.sample(n=n, random_state=42).reset_index(drop=True)\
            if n is not None else netflix.df
        
        print(f"Validating {n} samples from {validation_data} using {rulepath}")
        validator(netflix, rules, f"{data_label}-{rule_label}")
        sys.exit(0)
        
    
    # filepath = f"data/cidds_wk2_attack.csv"
    # filepath = f"data/cidds_wk2_all.csv"
    # filepath = f"data/cidds_wk3_all.csv"
    # cidds = Cidds001(filepath)
    # limit = 0
    
    filepath = f"data/netflix.csv"
    netflix = Netflix(filepath)
    
    if sys.argv[1] == 'all':
        # limit = cidds.df.shape[0]
        limit = netflix.df.shape[0]
        
        # main(cidds, sys.argv[1])
        # sys.exit(0)
    else:
        # cidds.df = cidds.df.sample(n=int(sys.argv[1]), random_state=42)
        limit = int(sys.argv[1])
    # refcidds = Cidds001("data/cidds_wk4_all.csv")
    # main(cidds, refcidds, limit)
    
    refnetflix = Netflix("data/netflix.csv")
    main(netflix, refnetflix, limit)
    sys.exit(0)