from mlxtend.frequent_patterns import apriori, fpgrowth, hmine, fpmax
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pandas as pd
from time import perf_counter
from typing import *
from collections import defaultdict
import sympy as sp

from anuta.constructor import Constructor
from anuta.known import *
from anuta.utils import log
from anuta.grammar import DomainType
from anuta.theory import Theory


def _encode_rule_pair(antecedent: FrozenSet[str], consequent: FrozenSet[str]) -> List[Tuple[str, FrozenSet[str]]]:
    # Encode antecedents
    predicates = frozenset(f"Eq({pred.split('_')[0]},{pred.split('_')[-1]})" for pred in antecedent)

    # Return list of (consequent_predicate, antecedents)
    return [(f"Eq({pred.split('_')[0]},{pred.split('_')[-1]})", predicates) for pred in consequent]

def get_missing_domain_rules(examples, domains) -> List[str]:
    """
    Generate rules for missing domains based on the provided examples and domains.
    """
    rules = []
    for varname, domain in domains.items():
        if domain.kind == DomainType.CATEGORICAL:
            # Check if the variable is present in the examples
            if varname not in examples.columns:
                continue
            
            # Get unique values in the column
            unique_values = set(examples[varname].unique())
            domain_values = set(domain.values)
            missing_values = domain_values - unique_values
            if missing_values:
                # Create rules for missing values
                for value in missing_values:
                    rule = f"Ne({varname},{value})"
                    rules.append(rule)
    return rules

class AsscoriationRuleLearner:
    def __init__(self, constructor: Constructor, algorithm='fpgrowth', 
                 limit=None, min_support=1e-10, **kwargs):
        self.algorithm = algorithm
        self.min_support = min_support
        self.kwargs = kwargs
        self.dataset = constructor.label
        #* Only support categorical variables.
        self.df = constructor.df[constructor.categoricals]
        if limit and limit < self.df.shape[0]:
            log.info(f"Limiting dataset to {limit} examples.")
            self.df = self.df.sample(n=limit, random_state=42)
            self.num_examples = limit
        else:
            self.num_examples = 'all'
        
        transactions = self.df.astype(str).apply(
            lambda row: [f"{col}_{val}" for col, val in row.items()], axis=1).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.df = pd.DataFrame(te_ary, columns=te.columns_)
        self.domains = constructor.anuta.domains
        self.learned_rules = []

    def learn(self, min_threshold=1) -> pd.DataFrame:
        log.info(f"Learning association rules ({self.algorithm}) from {len(self.df)} examples ...")
        method = None
        match self.algorithm:
            case 'apriori':
                method = apriori
            case 'fpgrowth':
                method = fpgrowth
            case 'hmine':
                method = hmine
            # case 'fpmax':
            #     method = fpmax
            case _:
                raise ValueError("Unsupported algorithm: {}".format(self.algorithm))
        
        start = perf_counter()
        frequent_itemsets= method(
            self.df, min_support=self.min_support, use_colnames=True, **self.kwargs)
        log.info(f"Frequent itemsets found: {len(frequent_itemsets)}")
        
        aruledf = association_rules(frequent_itemsets, 
                                    metric="confidence",
                                    #* Learn hard rules by default (min_threshold=1)
                                    min_threshold=min_threshold,) #, support_only=True
        log.info(f"Association rules found: {len(aruledf)}")
        end = perf_counter()
        log.info(f"Association rule learning took {end - start:.2f} seconds.")
        
        # self.learned_rules = self.extract_rules(aruledf)
        self.learned_rules = self.extract_rules_parallel(aruledf)
        log.info(f"Extracted {len(self.learned_rules)} rules.")
        
        assumptions = set()
        for varname, domain in self.domains.items():
            if domain.kind == DomainType.CATEGORICAL:
                assumptions.add(f"{varname} >= 0")
                assumptions.add(f"{varname} <= {max(domain.values)}")
        # assumptions = set(assumptions) | set(get_missing_domain_rules(self.df, self.domains))
        
        rules = set(self.learned_rules) | assumptions
        sprules = [sp.sympify(rule) for rule in rules]
        Theory.save_constraints(sprules, 
                                f'{self.algorithm}_{self.dataset}_{self.num_examples}.pl')
        
        return
    
    def extract_rules_parallel(self, aruledf: pd.DataFrame) -> List[str]:
        premisesmap = defaultdict(set)
        #* === Parallel rule encoding ===
        num_cores = cpu_count()
        futures = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for antecedent, consequent in zip(
                aruledf.antecedents.to_numpy(), 
                aruledf.consequents.to_numpy()
            ):
                futures.append(executor.submit(_encode_rule_pair, antecedent, consequent))

            for future in as_completed(futures):
                for conseq_predicate, predicates in future.result():
                    premisesmap[conseq_predicate].add(predicates)

        #* === Merge premises → consequents (serial logic remains) ===
        consequentsmap = defaultdict(set)
        for conseq_predicate, premise_sets in premisesmap.items():
            for premise in premise_sets:
                is_redundant = False
                for other in premise_sets - {premise}:
                    if premise > other:  # premise is a superset
                        is_redundant = True
                        break
                if not is_redundant:
                    consequentsmap[premise].add(conseq_predicate)

        #* === Format readable rules ===
        arules = []
        for antecedents, consequents in consequentsmap.items():
            premise = ' & '.join(antecedents)
            conclusion = ' & '.join(consequents)
            arule = f"({premise}) >> ({conclusion})"
            arules.append(arule)

        return arules
    
    def extract_rules(self, aruledf: pd.DataFrame) -> List[str]:
        premisesmap = defaultdict(set)

        # Build mapping from each consequent predicate to its supporting antecedent sets
        for antecedent, consequent in zip(
            aruledf.antecedents.to_numpy(), 
            aruledf.consequents.to_numpy()
        ):
            # Encode antecedents
            predicates = set()
            for predicate in antecedent:
                varname = predicate.split('_')[0]
                value = predicate.split('_')[-1]
                predicates.add(f"Eq({varname},{value})")
            predicates = frozenset(predicates)
            
            # Encode consequents and populate mapping
            for predicate in consequent:
                varname = predicate.split('_')[0]
                value = predicate.split('_')[-1]
                conseq_predicate = f"Eq({varname},{value})"
                premisesmap[conseq_predicate].add(predicates)

        # Merge premises → consequents
        consequentsmap = defaultdict(set)
        for conseq_predicate, premise_sets in premisesmap.items():
            for premise in premise_sets:
                is_redundant = False
                for other in premise_sets - {premise}:
                    if premise > other:  # premise is a superset
                        is_redundant = True
                        break
                if not is_redundant:
                    consequentsmap[premise].add(conseq_predicate)

        # Format readable rules
        arules = []
        for antecedents, consequents in consequentsmap.items():
            premise = ' & '.join(antecedents)
            conclusion = ' & '.join(consequents)
            arule = f"({premise}) >> ({conclusion})"
            arules.append(arule)

        return arules
    