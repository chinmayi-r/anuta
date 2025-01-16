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
from anuta.utils import FLAGS, log

    
def main(constructor: Constructor, refconstructor: Constructor, limit: int):
    if FLAGS.baseline:
        miner_valiant(constructor, limit)
    else:
        miner_versionspace(constructor, refconstructor, limit)

if __name__ == '__main__':
    assert '.csv' in FLAGS.data, "Data file is not CSV."
    data_label = FLAGS.data.split('/')[-1].split('.')[-2]
    dataset = FLAGS.dataset.lower()
    if dataset == 'cidds':
        constructor = Cidds001(FLAGS.data)
    elif dataset == 'netflix':
        constructor = Netflix(FLAGS.data)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if FLAGS.limit:
        limit = FLAGS.limit
        limit = int(limit[:-1]) * 1024 if 'k' in limit else int(limit)
        assert limit <= constructor.df.size, f"Dataset size {constructor.df.size} < {limit=}"
    else:
        limit = constructor.df.size
    
    #* Disable domain counting for baseline method.
    if FLAGS.baseline or not FLAGS.dc:
        FLAGS.config.DOMAIN_COUNTING = False
        
    if FLAGS.learn:
        refdata = FLAGS.ref
        refconstructor = None
        if not refdata:
            if dataset == 'cidds':
                refdata = "data/cidds_wk4_all.csv"
                refconstructor = Cidds001(refdata)
            elif dataset == 'netflix':
                refdata = "data/netflix.csv"
                refconstructor = Netflix(refdata)
        else:
            refconstructor = Cidds001(refdata) if dataset == 'cidds' else Netflix(refdata)
            
        log.info(f"Learning from {limit} examples in {FLAGS.data}")
        log.info(f"Domain counting enabled: {FLAGS.dc}")
        log.info(f"Using baseline method: {FLAGS.baseline}")
        log.info(f"Reference data: {refdata}")     
        
        main(constructor, refconstructor, limit)

    elif FLAGS.validate:
        constructor.df = constructor.df.sample(n=limit, random_state=42).reset_index(drop=True) \
            if FLAGS.limit else constructor.df
        
        assert FLAGS.rules, "No rules file provided."
        rulepath = FLAGS.rules
        assert rulepath.endswith('.rule'), "Invalid rule file."
        rules = Theory.load_constraints(rulepath, False)
        rule_label = int(rulepath.split('_')[1])
        label = f"{data_label}-{rule_label}"
        
        log.info(f"Validating {limit} samples from {FLAGS.data} using {rulepath}")
        
        validator(constructor, rules, label)
    
    sys.exit(
        pprint("Configuration:", dict(FLAGS.config), sep='\n')
    )