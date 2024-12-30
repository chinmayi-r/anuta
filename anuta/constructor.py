from dataclasses import dataclass
from rich import print as pprint
from collections import defaultdict
import pandas as pd
import numpy as np
import sympy as sp
import json
import sys

from anuta.grammar import AnutaMilli, Bounds, Anuta, Domain, DomainType, ConstantType, Constants
from anuta.known import *
from anuta.utils import log, save_constraints, to_big_camelcase


@dataclass
class DomainCounter:
    count: int
    frequency: int
    #* Lexicographic ordering
    def __lt__(self, other):  # Less than
        if self.count == other.count:
            return self.frequency < other.frequency
        else:
            return self.count < other.count

    def __le__(self, other):  # Less than or equal to
        if self.count == other.count:
            return self.frequency <= other.frequency
        else:
            return self.count <= other.count

    def __eq__(self, other):  # Equal to
        if self.count == other.count:
            return self.frequency == other.frequency
        else:
            return self.count == other.count
        
    def __gt__(self, other):  # Greater than
        if self.count == other.count:
            return self.frequency > other.frequency
        else:
            return self.count > other.count

    def __ge__(self, other):  # Greater than or equal to
        if self.count == other.count:
            return self.frequency >= other.frequency
        else:
            return self.count >= other.count
    
    def __repr__(self) -> str:
        return f"(count:{self.count} freq:{self.frequency})"
    
    def __str__(self) -> str:
        return self.__repr__()

class Constructor(object):
    def __init__(self) -> None:
        self.df: pd.DataFrame = None
        self.anuta: AnutaMilli | Anuta = None
    
    def get_indexset_and_counter(
            self, df: pd.DataFrame,
            domains: dict[str, Domain],
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, DomainCounter]]:
        pass
        
class Cidds001(Constructor):
    def __init__(self, filepath) -> None:
        log.info(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath)
        #* Discard the timestamps for now, and Flows is always 1.
        self.df = self.df.drop(columns=['Date first seen', 'Flows'])
        col_to_var = {col: to_big_camelcase(col) for col in self.df.columns}
        self.df.rename(columns=col_to_var, inplace=True)
        variables = list(self.df.columns)
        
        #* Convert the Flags and Proto columns to integers        
        self.df['Flags'] = self.df['Flags'].apply(flag_map)
        self.df['Proto'] = self.df['Proto'].apply(proto_map)
        for var in ['SrcIpAddr', 'DstIpAddr']:
            self.df[var] = self.df[var].apply(ip_map)
        self.categorical = cidds_categorical
        
        domains = {}
        for var in self.df.columns:
            if var not in self.categorical:
                domains[var] = Domain(DomainType.NUMERICAL, 
                                      Bounds(self.df[var].min().item(), 
                                             self.df[var].max().item()), 
                                      None)
            else:
                domains[var] = Domain(DomainType.CATEGORICAL, 
                                      None, 
                                      self.df[var].unique())
        
        #* Add the constants associated with the vars.
        prior_kb = []
        self.constants: dict[str, Constants] = {}
        for var in variables:
            if 'ip' in var.lower():
                #& Don't need to add the IP constants here, as the domain is small and can be enumerated.
                self.constants[var] = Constants(
                    kind=ConstantType.ASSIGNMENT,
                    values=cidds_constants['ip']
                )
            elif 'pt' in var.lower():
                self.constants[var] = Constants(
                    kind=ConstantType.ASSIGNMENT,
                    values=cidds_constants['port']
                )
            elif 'packet' in var.lower():
                self.constants[var] = Constants(
                    kind=ConstantType.SCALAR,
                    #* Sort the values in ascending order.
                    values=sorted(cidds_constants['packet'])
                )
                
        self.anuta = Anuta(variables, domains, self.constants, prior_kb)
        pprint(self.anuta.variables)
        pprint(self.anuta.domains)
        pprint(self.anuta.constants)
        print(f"Prior KB size: {len(self.anuta.prior_kb)}:")
        print(f"\t{self.anuta.prior_kb}\n")
        
        # save_constraints(self.anuta.initial_kb + self.anuta.prior_kb, 'initial_constraints_arity3_negexpr')
        print(f"Initial KB size: {len(self.anuta.initial_kb)}")
        return
    
    def get_indexset_and_counter(
            self, df: pd.DataFrame,
            domains: dict[str, Domain],
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, DomainCounter]]:
        indexset = {}
        #^ dict[var: dict[val: array(indices)]] // Var -> {Distinct value -> indices}
        for cat in self.categorical:
            indices = df.groupby(by=cat).indices
            filtered_indices = {}
            if cat in self.constants:
                for val in indices:
                    if val in self.constants[cat].values:
                        filtered_indices[val] = indices[val]
            indexset[cat] = filtered_indices
        #TODO: Create index set also for numerical variables.
        
        fcount = defaultdict(dict)
        #^ dict[var: dict[val: count]] // Var -> {Value of interest -> (count,freq)}
        for cat in indexset:
            if cat in self.constants:
                #* Don't enumerate the domain if it has associated constants.
                values = self.constants[cat].values
            else:
                values = indexset[cat].keys()
            
            for val in values:
                #! Some values could be in the constants but not in the data (partition).
                if val in domains[cat].values and val in indexset[cat]:
                    #* Initialize the counters to the frequency of the value in the data.
                    #& Prioritize rare values (inductive bias).
                    #& Using the frequency only as a tie-breaker [count, freq].
                    freq = len(indexset[cat][val])
                    dc = DomainCounter(count=0, frequency=freq)
                    fcount[cat] |= {val: dc} if type(val) == int else {val.item(): dc}
        # for varname, dc in fcounts.items():
        # domain = anuta.domains.get(varname)
        # assert domain.kind == Kind.CATEGORICAL, "Found numerical variable in DC counter."
        
        # filtered_dc = {}
        # for key in dc:
        #     if key in domain.values:
        #         filtered_dc |= dc[key]
        # fcounts[varname] = filtered_dc
        return indexset, fcount
        


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