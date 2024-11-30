from dataclasses import dataclass
from rich import print as pprint
import pandas as pd
import json
import sys

from grammar import Anuta, Bounds


class Constructor(object):
    def __init__(self) -> None:
        self.df: pd.DataFrame = None
        self.anuta: Anuta = None

class Millisampler(Constructor):
    def __init__(self, filepath: str) -> None:
        boundsfile = f"./data/meta_bounds.json"
        print(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath)
        
        variables = []
        for col in self.df.columns:
            if col not in ['server_hostname', 'window', 'stride']:
                # if len(col.split('_')) > 1 and col.split('_')[1].isdigit(): continue
                variables.append(col)
        constants = {
            'burst_threshold': round(2891883 / 7200), # round(0.5*metadf.ingressBytes_sampled.max().item()),
        }
        
        canaries = {
            'canary_max10': (0, self.df.ingressBytes_aggregate.max().item()),
            #^ Max(u1, u2, ..., u10) == canary_max10
            'canary_premise': (0, 1),
            'canary_conclusion': (constants['burst_threshold']+1, constants['burst_threshold']+1),
            #^ (canary_premise > 0) => (canary_max10 + 1 ≥ burst_threshold)
        }
        variables.extend(canaries.keys())

        #* Load the bounds directly from the file
        with open(boundsfile, 'r') as f:
            bounds = json.load(f)
            bounds = {k: Bounds(v[0], v[1]) for k, v in bounds.items()}
        # bounds = {}
        # for col in metadf.columns:
        #     if col in ['server_hostname', 'window', 'stride']: 
        #         continue
        #     bounds[col] = Bounds(metadf[col].min().item(), metadf[col].max().item())
        for n, c in constants.items():
            bounds[n] = Bounds(c, c)
        for n, c in canaries.items():
            bounds[n] = Bounds(c[0], c[1])
        
        self.anuta = Anuta(variables, bounds, constants, operators=[0, 1, 2])
        pprint(self.anuta.variables)
        pprint(self.anuta.constants)
        pprint(self.anuta.bounds)
        
        self.anuta.populate_kb()
        return