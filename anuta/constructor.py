from dataclasses import dataclass
from rich import print as pprint
import pandas as pd
import numpy as np
import sympy as sp
import json
import sys

from grammar import AnutaMilli, Bounds, Anuta, Domain, Kind, Constant
from known import ip_map, cidds_constants, flag_map, proto_map, cidds_ip_conversion
from utils import log, save_constraints


class Constructor(object):
    def __init__(self) -> None:
        self.df: pd.DataFrame = None
        self.anuta: AnutaMilli | Anuta = None
        
class Cidds001(Constructor):
    def __init__(self, filepath) -> None:
        log.info(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath)
        #* Discard the timestamps for now, and Flows is always 1.
        self.df = self.df.drop(columns=['Date first seen', 'Flows'])
        #* Convert the Flags and Proto columns to integers        
        self.df['Flags'] = self.df['Flags'].apply(flag_map)
        self.df['Proto'] = self.df['Proto'].apply(proto_map)
        
        col_to_var = {col: col.replace(' ', '_') for col in self.df.columns}
        variables = list(col_to_var.values())
        #? Should port be a categorical variable? Sometimes we need range values (i.e., application and dynamic ports).
        categorical = ['Flags', 'Proto', 'Src IP Addr', 'Dst IP Addr'] + ['Src Pt', 'Dst Pt']
        
        domains = {}
        for col in self.df.columns:
            var = col_to_var[col]
            if col not in categorical:
                domains[var] = Domain(Kind.NUMERICAL, Bounds(self.df[col].min().item(), self.df[col].max().item()), None)
            else:
                #& Apply domain knowledge
                if 'Addr' in col:
                    self.df[col] = self.df[col].apply(ip_map)
                domains[var] = Domain(Kind.CATEGORICAL, None, self.df[col].unique())
        
        prior_kb = []
        constants = {}
        for var in variables:
            if 'IP' in var:
                #& Don't need to add the IP constants here, as the domain is small and can be enumerated.
                # constants[var] = Constant.ASSIGNMENT
                # constants[var].values = cidds_constants['ip']
                #* Add the prior knowledge
                for ip in cidds_ip_conversion.values():
                    if not ip in domains[var].values:
                        prior_kb.append(sp.Ne(sp.symbols(var), sp.S(ip)))
            elif 'Pt' in var:
                constants[var] = Constant.ASSIGNMENT
                constants[var].values = cidds_constants['port']
            elif 'Packet' in var:
                constants[var] = Constant.SCALAR
                constants[var].values = cidds_constants['packet']
                
        self.anuta = Anuta(variables, domains, constants, prior_kb)
        pprint(self.anuta.variables)
        pprint(self.anuta.domains)
        pprint(self.anuta.constants)
        print(f"Prior KB size: {len(self.anuta.prior_kb)}:")
        print(f"\t{self.anuta.prior_kb}\n")
        
        self.anuta.populate_kb()
        # pprint(self.anuta.initial_kb)
        # save_constraints(self.anuta.initial_kb + self.anuta.prior_kb, 'initial_constraints_arity3_negexpr')
        print(f"Initial KB size: {len(self.anuta.initial_kb)}")
        return
        


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
        
        self.anuta = AnutaMilli(variables, bounds, constants, operators=[0, 1, 2])
        pprint(self.anuta.variables)
        pprint(self.anuta.constants)
        pprint(self.anuta.bounds)
        
        self.anuta.populate_kb()
        return