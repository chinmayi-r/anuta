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

from grammar import Anuta
from constructor import Constructor, Millisampler
from miner import millisampler_miner
from utils import log, save_constraints
import json


anuta : Anuta = None

    
def main(constructor: Constructor, label: str):
    millisampler_miner(constructor, label)
    

if __name__ == '__main__':
    boundsfile = f"./data/meta_bounds.json"
    filepath = f"./data/meta_w10_s5_{sys.argv[1]}.csv"
    
    millisampler = Millisampler(filepath)
    
    main(millisampler, sys.argv[1])
    sys.exit(0)