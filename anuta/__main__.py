from pathlib import Path
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

from anuta.constructor import Constructor, Millisampler, Cidds001, Netflix, Cicids2017, Yatesbury
from anuta.tree import EntropyTreeLearner, XgboostTreeLearner, LightGbmTreeLearner
from anuta.association import AsscoriationRuleLearner
from anuta.theory import Theory
from anuta.miner import miner_versionspace, miner_valiant, validator
from anuta.utils import log
from anuta.cli import FLAGS

    
def main(constructor: Constructor, refconstructor: Constructor, limit: int):
    if FLAGS.tree:
        match FLAGS.tree:
            case 'dt':
                learner = EntropyTreeLearner(constructor, limit=limit)
                log.info("Learning constraints using normal decision tree...")
            case 'xgb':
                learner = XgboostTreeLearner(constructor, limit=limit)
                log.info("Learning constraints using XGBoost...")
            case 'lgbm':
                learner = LightGbmTreeLearner(constructor, limit=limit)
                log.info("Learning constraints using LightGBM...")
            case _:
                raise ValueError(f"Unknown tree type: {FLAGS.tree}")
        learner.learn()
    elif FLAGS.assoc:
        learner = AsscoriationRuleLearner(
            constructor, algorithm=FLAGS.assoc, limit=limit)
        log.info("Learning constraints using association rules...")
        learner.learn()
    elif FLAGS.baseline:
        miner_valiant(constructor, limit)
    else:
        miner_versionspace(constructor, refconstructor, limit)

if __name__ == '__main__':
    FLAGS(sys.argv)
    
    assert '.csv' in FLAGS.data, "Data file is not CSV."
    data_label = FLAGS.data.split('/')[-1].split('.')[-2]
    dataset = FLAGS.dataset.lower()
    match dataset:
        case 'cidds':
            constructor = Cidds001(FLAGS.data)
        case 'netflix':
            constructor = Netflix(FLAGS.data)
        case 'cicids':
            constructor = Cicids2017(FLAGS.data)
        case 'yatesbury':
            constructor = Yatesbury(FLAGS.data)
        case 'metadc':
            constructor = Millisampler(FLAGS.data)
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    if FLAGS.limit:
        limit = FLAGS.limit
        limit = int(limit[:-1]) * 1024 if 'k' in limit else int(limit)
        assert limit <= constructor.df.shape[0], f"Dataset size {constructor.df.size} < {limit=}"
    else:
        limit = constructor.df.shape[0]
    
    #* Disable domain counting for baseline method.
    if FLAGS.baseline or not FLAGS.dc:
        FLAGS.config.DOMAIN_COUNTING = False
        
    if FLAGS.learn:
        refdata = FLAGS.ref
        refconstructor = None
        if not FLAGS.tree and not FLAGS.assoc:
            if not refdata:
                match dataset:
                    case 'cidds':
                        refdata = "data/cidds_wk4_all.csv"
                        refconstructor = Cidds001(refdata)
                    case 'netflix':
                        refdata = "data/netflix.csv"
                        refconstructor = Netflix(refdata)
                    case 'cicids':
                        refdata = "data/cicids_friday_normal.csv"
                        refconstructor = Cicids2017(refdata)
            else:
                refconstructor = Cidds001(refdata) \
                    if dataset == 'cidds' else Netflix(refdata)
            
        log.info(f"Learning from {limit} examples in {FLAGS.data}")
        log.info(f"Domain counting enabled: {FLAGS.config.DOMAIN_COUNTING}")
        log.info(f"Using baseline method: {FLAGS.baseline}")
        if refconstructor:
            log.info(f"Reference data: {refdata}")     
        
        main(constructor, refconstructor, limit)

    elif FLAGS.validate:
        constructor.df = constructor.df.sample(n=limit, random_state=42).reset_index(drop=True) \
            if FLAGS.limit else constructor.df
        
        assert FLAGS.rules, "No rules file provided."
        rulepath = FLAGS.rules
        assert rulepath.endswith('.pl'), "Invalid rule file."
        rules = Theory.load_constraints(rulepath, False)
        rule_label = "_".join(rulepath.split('_')[1:])[:-3]
        checked = 'checked' in rulepath
        label = f"{data_label}-{rule_label}"
        
        log.info(f"Validating {len(constructor.df)} samples from {FLAGS.data} using {rulepath}")
        
        violation_rate, _ = validator(constructor, rules, label)
        violation_record = ','.join([data_label, str(rule_label), str(checked), str(violation_rate)])
        #TODO: Specify all file names in the config file.
        violation_file = f"violation.csv"
        if Path(violation_file).exists():
            with open(violation_file, 'a') as f:
                f.write(violation_record + '\n')
        else:
            with open(violation_file, 'w') as f:
                f.write("data,rule,checked,violation_rate\n")
                f.write(violation_record + '\n')

    sys.exit(
        pprint("Configuration:", dict(FLAGS.config), sep='\n')
    )