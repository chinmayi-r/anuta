import math
import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.tree import H2OTree, H2OLeafNode, H2OSplitNode
from typing import *
import itertools
from collections import defaultdict
from time import perf_counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import sympy as sp
from rich import print as pprint
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

from anuta.constructor import Constructor, Cidds001
from anuta.theory import Theory
from anuta.known import cidds_categoricals
from anuta.utils import log, to_big_camelcase


class EntropyTreeLearner:
    """Tree learner based on information gain, using H2O's implementation."""
    def __init__(self, constructor: Constructor, limit=None):
        h2o.init(nthreads=-1)  # -1 = use all available cores
        h2o.no_progress()  # Disables all progress bar output
        
        if limit and limit < constructor.df.shape[0]:
            log.info(f"Limiting dataset to {limit} examples.")
            constructor.df = constructor.df.sample(n=limit, random_state=42)
            self.num_examples = limit
        else:
            self.num_examples = constructor.df.shape[0]
            
        self.dataset = constructor.label
        match constructor.label:
            case 'cidds':
                constructor: Cidds001 = constructor
                constructor.df[constructor.categoricals] = \
                    constructor.df[constructor.categoricals].astype('category')
                self.examples: h2o.H2OFrame = h2o.H2OFrame(constructor.df)
                self.categoricals = constructor.categoricals
                self.examples[self.categoricals] = self.examples[self.categoricals].asfactor()
            case _:
                #TODO: Add support for other datasets
                raise ValueError(f"Unsupported constructor: {constructor.label}")

        # variables: List[str] = self.examples.columns
        variables: List[str] = [
            var for var in self.examples.columns 
            # if var in self.categoricals
        ]
        self.featuregroups = defaultdict(list)
        for target in variables:
            features = [v for v in variables if v != target]
            for n in range(1, len(features)+1):
                self.featuregroups[target] += itertools.combinations(features, n)
        
        self.model_configs = {}
        self.model_configs['classification'] = dict(
            # model_id="clf_tree",
            ntrees=1,                 # Build only one tree
            max_depth=len(variables),
            min_rows=1,               # Minimum number of observations in a leaf
            min_split_improvement=1e-6,
            sample_rate=1.0,          # Use all rows
            mtries=-2,                # Use all features (set to -2 for all features)
            seed=42,                  # For reproducibility
            categorical_encoding="Enum"  # Native handling of categorical features
        )
        self.model_configs['regression'] = dict(
            # model_id="reg_tree",
            ntrees=1,                 # Build only one tree
            max_depth=len(variables)//2, #TODO: To be tuned
            min_rows=100,             #TODO: To be tuned
            sample_rate=1.0,          # Use all rows
            mtries=-2,                # Use all features (set to -2 for all features)
            seed=42,                  # For reproducibility
            categorical_encoding="Enum"  # Native handling of categorical features
        )
        
        # self.domains = constructor.anuta.domains
        self.domains = {}
        for varname in variables:
            if varname in self.categoricals: 
                self.domains[varname] = sorted(list(constructor.df[varname].unique()))
            else:
                self.domains[varname] = (
                    self.examples[varname].min(), 
                    self.examples[varname].max()
                )
        #* dTypes: {'int', 'real', 'enum'(categorical)}
        self.dtypes = {varname: t for varname, t in self.examples.types.items()}
        self.trees: Dict[str, List[H2ORandomForestEstimator]] = defaultdict(list)
        self.learned_rules: Set[str] = set()
        pprint(self.domains)
        pprint(self.dtypes)
    
    def learn(self):
        total_trees = len(self.featuregroups) * \
            len(self.featuregroups[list(self.featuregroups)[0]])
        log.info(f"Learning {total_trees} trees from {len(self.examples)} examples.")
        
        start = perf_counter()
        treeid = 1
        for target, feature_group in self.featuregroups.items():
            log.info(f"Learning trees for {target} with {len(feature_group)} feature groups.")
            if target in self.categoricals:
                params = self.model_configs['classification']
            else:
                params = self.model_configs['regression']
            
            for i, features in enumerate(feature_group):
                model_id = f"{target}_tree_{i+1}"
                params['model_id'] = model_id
                dtree = H2ORandomForestEstimator(**params)
                dtree.train(x=list(features), y=target, training_frame=self.examples)  
                self.trees[target].append(dtree)
                print(f"... Trained {treeid}/{total_trees} ({treeid/total_trees:.1%}) trees ({target=}).", end='\r')
                treeid += 1
        end = perf_counter()
        log.info(f"Training {total_trees} trees took {end - start:.2f} seconds.")
        
        start = perf_counter()
        self.learned_rules = self.extract_rules_from_treepaths()
        end = perf_counter()
        log.info(f"Learned {len(self.learned_rules)} rules from {total_trees} trees.")
        log.info(f"Extracting rules took {end - start:.2f} seconds.")
        
        assumptions = set()
        for varname, domain in self.domains.items():
            if varname in self.categoricals:
                assumptions.add(f"{varname} >= 0")
                assumptions.add(f"{varname} <= {max(domain)}")
            else:
                assumptions.add(f"{varname} >= {domain[0]}")
                assumptions.add(f"{varname} <= {domain[1]}")
        
        rules = self.learned_rules | assumptions
        sprules = [sp.sympify(rule) for rule in rules]
        Theory.save_constraints(sprules, f'dtree_{self.dataset}_{self.num_examples}.pl')
        
        #TODO: Interpret rules with domains to enable `X [opt] s•Y` rules.
        return
        
    def extract_rules_from_treepaths(self) -> Set[str]:
        learned_rules = set()
        for target, all_treepaths in self.extract_paths_from_all_trees().items():
            total_rules = 0
            for treeidx, pathconditions in enumerate(all_treepaths):
                if not pathconditions: 
                    #* No valid paths found in this tree
                    continue 
                
                if target in self.categoricals:
                    ruleset = defaultdict(set)
                else:
                    ruleset = defaultdict(dict)
                    '''Collect leaf ranges for regression trees'''
                    dtree: H2ORandomForestEstimator = self.trees[target][treeidx]
                    leaf_assignments = dtree.predict_leaf_node_assignment(self.examples)
                    #* Bind leaf assignments with original target column
                    hf_leaf = self.examples.cbind(leaf_assignments)
                    leaf_col = leaf_assignments.columns[0]
                    
                    #* Group by leaf and compute min/max of the target variable
                    leaf_stats = hf_leaf.group_by(leaf_col).min(target).max(target).get_frame()
                    leaf_stats.set_names(['leaf_id', 'leaf_min', 'leaf_max'])
                    leaf_stats_df = leaf_stats.as_data_frame(use_multi_thread=True)
                    leaf_id_col = leaf_stats_df.columns[0]      
                    min_col = leaf_stats_df.columns[1]          
                    max_col = leaf_stats_df.columns[2]          
                    leaf_ranges = {
                        row[leaf_id_col]: {'min': row[min_col], 'max': row[max_col]}
                        for _, row in leaf_stats_df.iterrows()
                    }

                for targetcls, records in pathconditions.items():
                    for record in records:
                        merged_conditions = {}
                        for condition in record['conditions']:
                            varname, op, varval = condition.split('_')
                            varval = eval(varval)
                            if varname not in merged_conditions:
                                merged_conditions[varname] = defaultdict(None)

                            if op in ['∈', '∉']:
                                values = merged_conditions[varname].get(op, set())
                                merged_conditions[varname][op] = values | set(varval)
                            elif op == '>':  
                                if self.dtypes[varname] == 'int':
                                    varval = math.floor(varval)
                                value = merged_conditions[varname].get(op, float('+inf'))
                                merged_conditions[varname][op] = min(value, varval)
                            elif op == '≤':
                                if self.dtypes[varname] == 'int':
                                    varval = math.ceil(varval)
                                value = merged_conditions[varname].get(op, float('-inf'))
                                merged_conditions[varname][op] = max(value, varval)

                        predicates = []
                        for varname, conditions in merged_conditions.items():
                            operators = conditions.keys()
                            if set(['∈', '∉']) & operators:
                                #* Categorical var
                                assert not set(['≤', '>']) & operators, f"{varname} has mixed {conditions=}"
                                _invals = conditions.get('∈', set())
                                _outvals = conditions.get('∉', set())
                    
                                invals = _invals - _outvals
                                outvals = _outvals - _invals
                                domain = set(self.domains[varname])

                                #* Use the most succinct representation
                                if invals:
                                    diffvals = domain - invals
                                    if len(diffvals) < len(invals):
                                        outvals |= diffvals
                                        invals = set()
                                if outvals:
                                    diffvals = domain - outvals
                                    if len(diffvals) < len(outvals):
                                        invals |= diffvals
                                        outvals = set()
                                
                                predicate = ''
                                for val in invals:
                                    predicate += f"Eq({varname}, {val})|"
                                predicate = predicate[:-1]
                                if len(invals) > 1:
                                    predicate = '( ' + predicate + ' )'
                                if predicate:
                                    predicates.append(predicate)
                    
                                for val in outvals:
                                    predicates.append(f"Ne({varname}, {val})")
                            else:
                                assert set(['≤', '>']) & operators, f"{varname} has no recognizd {conditions=}"
                                valmin = conditions.get('>', float('-inf'))
                                valmax = conditions.get('≤', float('+inf'))
                                if valmin >= valmax:
                                    #TODO: Accumulate logits to decide which condition to take. Discard all together for now.
                                    print(f"[Conflicting condition!!!]: {varname=}:{conditions}")
                                else:
                                    if valmin > float('-inf'):
                                        predicates.append(f"({varname} > {valmin})")
                                    if valmax < float('+inf'):
                                        predicates.append(f"({varname} <= {valmax})")

                        if predicates:
                            premise = ' & '.join(predicates)
                            if target in self.categoricals:
                                conclusion = f"Eq({target}, {targetcls})"
                                ruleset[premise].add(conclusion)
                            else:
                                leafid = record['pathid'].split('-')[1]
                                leafmin = leaf_ranges[leafid]['min']
                                leafmax = leaf_ranges[leafid]['max']
                                ruleset[premise]['min'] = min(
                                    ruleset[premise].get('min', float('+inf')), 
                                    leafmin
                                )
                                ruleset[premise]['max'] = max(
                                    ruleset[premise].get('max', float('-inf')), 
                                    leafmax
                                )

                rules = set()
                for premise, conclusions in ruleset.items():
                    if target in self.categoricals:
                        conclusion = '( ' + '&'.join(conclusions) + ' )' \
                            if len(conclusions) > 1 else conclusions.pop()
                    else:
                        targetmin = ruleset[premise]['min']
                        targetmax = ruleset[premise]['max']
                        conclusion = f"(({target}>={targetmin}) & ({target}<={targetmax}))"\
                            if targetmin != targetmax else f"Eq({target}, {targetmin})"
                    
                    rule = f"({premise}) >> {conclusion}"
                    rules.add(rule)
                total_rules += len(rules)
                learned_rules |= rules
            log.info(f"Extracted {len(rules)} rules from trees of {target}.")
        log.info(f"Total rules extracted: {len(learned_rules)}")
        return learned_rules
    
    def extract_paths_from_all_trees(self):
        """Extract path conditions from all trees."""
        all_tree_paths = defaultdict(list)
        for target, trees in self.trees.items():
            for treeidx, dtree in enumerate(tqdm(
                trees, desc=f"Extracting paths from trees of {target}"
            )):
                # print(f"Features: {self.featuregroups[target][treeidx]}")
                paths = self.extract_tree_paths(dtree, target)
                #* Paths could be empty `{}`, but keep it 
                #*  to match the indexing of `self.trees` to `all_tree_paths`
                all_tree_paths[target].append(paths)

        #* {target: [{tree1_paths}, {tree2_paths}, ...]}
        return all_tree_paths
    
   
    def extract_tree_paths(self, dtree: H2ORandomForestEstimator, target):
        MIN_LOGIT = 1-1e-6 if target in self.categoricals else 0
        treeinfo = dtree._model_json['output']['model_summary']
        assert treeinfo['number_of_trees'][0] == 1, f"H2O random forest has {treeinfo['number_of_trees']}>1 tree."
        
        logits = set()
        def recurse(node: H2OLeafNode|H2OSplitNode, path, path_suffix, tree_index):
            if node.__class__.__name__ == 'H2OLeafNode':
                leaf_id = path_suffix or '0'
                #* Final value is a probability (unlike boosting trees that use logits). 
                #* Only take pure leaves
                logits.add(node.prediction)
                if node.prediction > MIN_LOGIT:
                    paths.append({
                        'pathid': f"{tree_index}-{leaf_id}",
                        'logit': node.prediction,
                        'conditions': path.copy()
                    })
                return

            varname = node.split_feature

            # Handle categorical splits
            if node.left_levels or node.right_levels:
                left_categories = sorted([int(v) for v in node.left_levels])
                right_categories = sorted([int(v) for v in node.right_levels])
                if node.left_levels:
                    cond_left = f"{varname}_∈_{left_categories}"
                    cond_right = f"{varname}_∉_{left_categories}"
                else:
                    cond_left = f"{varname}_∉_{right_categories}"
                    cond_right = f"{varname}_∈_{right_categories}"
            else:
                # Numeric split
                cond_left = f"{varname}_≤_{node.threshold}"
                cond_right = f"{varname}_>_{node.threshold}"

            recurse(node.left_child, path + [cond_left], path_suffix + "L", tree_index)
            recurse(node.right_child, path + [cond_right], path_suffix + "R", tree_index)

        treepaths = {}
        tree_classes = dtree._model_json['output']['domains'][-1]
        # print(f"{target=} {tree_classes=}, variables={dtree._model_json['output']['names']}")
        if tree_classes is not None and len(tree_classes) > 2:
            #* Multi-class classification tree
            for targetcls in tree_classes:
                #! Assume always use one tree (`tree_number=0`) with RF
                htree = H2OTree(model=dtree, tree_number=0, tree_class=targetcls)
                paths = []
                recurse(htree.root_node, [], "", targetcls)
                if paths:
                    treepaths[targetcls] = paths
        else:
            #* Binomial or regression tree
            htree = H2OTree(model=dtree, tree_number=0)
            paths = []
            recurse(htree.root_node, [], "", 0)
            if paths:
                treepaths[0] = paths
        
        # print(logits)
        return treepaths
 

class XgboostTreeLearner:
    """Tree learner based on XGBoost."""
    def __init__(self, constructor: Constructor, limit=None):
        if limit and limit < constructor.df.shape[0]:
            log.info(f"Limiting dataset to {limit} examples.")
            constructor.df = constructor.df.sample(n=limit, random_state=42)
            self.num_examples = limit
        else:
            self.num_examples = constructor.df.shape[0]
            
        self.dataset = constructor.label
        match constructor.label:
            case 'cidds':
                constructor: Cidds001 = constructor
                self.examples = constructor.df.copy()
                self.examples[constructor.categoricals] = \
                    self.examples[constructor.categoricals].astype('category')
                self.categoricals = constructor.categoricals
            case _:
                raise ValueError(f"Unsupported constructor: {constructor.label}")
        
        variables: List[str] = [
            var for var in self.examples.columns
            # if var in self.categoricals
        ] 
        self.featuregroups = defaultdict(list)
        for target in variables:
            features = [v for v in variables if v != target]
            for n in range(1, len(features)+1):
                self.featuregroups[target] += itertools.combinations(features, n)
        
        common_config = dict(
                min_child_weight=0,    # small → allows fine splits
                gamma=0,               # allow all positive-gain splits
                grow_policy='depthwise',  # ensures full-depth growth
                # # tree_method='exact',   # for most deterministic behavior
                # # subsample=1,
                # # colsample_bytree=1,
                learning_rate=1,        # set high so pure leaves dominate logits
                n_estimators=1, 
                reg_alpha=0,   # L1 regularization term on weights
                reg_lambda=1,   # L2 regularization term on weights
                enable_categorical=True)
        self.model_configs = {}
        self.model_configs['classification'] = dict(
            objective = 'multi:softprob',
            max_depth=len(variables), # high enough to split until pure
            **common_config,
        )
        self.model_configs['regression'] = dict(
            objective = 'reg:squarederror',
            max_depth=len(variables)//2, #TODO: To be tuned
            **common_config,
        )
                
        #TODO: Unify `Domain`
        self.domains = {}
        for varname in variables:
            if varname in self.categoricals: 
                self.domains[varname] = sorted(
                    [n.item() for n in constructor.df[varname].unique()])
            else:
                self.domains[varname] = (
                    self.examples[varname].min().item(), 
                    self.examples[varname].max().item(),
                )
        self.dtypes = {}
        #TODO: Unify `DomainType`
        #* dTypes: {'int', 'real', 'enum'(categorical)}
        for varname, dtype in self.examples.dtypes.items():
            if dtype.name == 'category':
                self.dtypes[varname] = 'enum'
            elif dtype.name in ['int64', 'int32']:
                self.dtypes[varname] = 'int'
            elif dtype.name in ['float64', 'float32']:
                self.dtypes[varname] = 'real'
            else:
                raise ValueError(f"Unsupported data type {dtype} for variable {varname}.")
        self.trees: Dict[str, List[XGBClassifier|XGBRegressor]] = defaultdict(list)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.learned_rules: Set[str] = set()
        pprint(self.domains)
        pprint(self.dtypes)
    
    def learn(self):
        total_trees = len(self.featuregroups) * \
            len(self.featuregroups[list(self.featuregroups)[0]])
        log.info(f"Learning {total_trees} trees from {len(self.examples)} examples.")
        
        start = perf_counter()
        treeid = 1
        for target, feature_group in self.featuregroups.items():
            log.info(f"Learning trees for {target} with {len(feature_group)} feature groups.")
            modelcls = None
            if target in self.categoricals:
                params = self.model_configs['classification']
                modelcls = XGBClassifier
            else:
                params = self.model_configs['regression']
                modelcls = XGBRegressor
            num_class = len(self.domains[target]) if target in self.categoricals else 1
            
            y = self.examples[target]
            if target in self.categoricals:
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                self.label_encoders[target] = encoder
            
            for i, features in enumerate(feature_group):
                model = modelcls(num_class=num_class, **params)
                X = self.examples[list(features)]
                model.fit(X, y)
                self.trees[target].append(model)
                print(f"... Trained {treeid}/{total_trees} ({treeid/total_trees:.1%}) trees ({target=}).", end='\r')
                treeid += 1
        end = perf_counter()
        log.info(f"Training {total_trees} trees took {end - start:.2f} seconds.")
        
        start = perf_counter()
        self.learned_rules = self.extract_rules_from_pathconditions()
        end = perf_counter()
        log.info(f"Learned {len(self.learned_rules)} rules from {total_trees} trees.")
        log.info(f"Extracting rules took {end - start:.2f} seconds.")
        
        assumptions = set()
        for varname, domain in self.domains.items():
            if varname in self.categoricals:
                assumptions.add(f"{varname} >= 0")
                assumptions.add(f"{varname} <= {max(domain)}")
            else:
                assumptions.add(f"{varname} >= {domain[0]}")
                assumptions.add(f"{varname} <= {domain[1]}")
        rules = self.learned_rules | assumptions
        sprules = [sp.sympify(rule) for rule in rules]
        Theory.save_constraints(sprules, f'xgb_{self.dataset}_{self.num_examples}.pl')
        
        return
    
    def extract_rules_from_pathconditions(self) -> Set[str]:
        learned_rules = set()
        for target, pathconditions in self.extract_conditions_from_treepaths().items():
            targetvar = target
            rules = set()
            for targetcls, conditions in pathconditions.items():
            # tqdm(
            #     pathconditions.items(), total=len(pathconditions),
            #     desc=f"Extracting rules from {target} conditions"
            # ):
                for record in conditions:
                    var_conditions = {}
                    for condition in record['conditions']:
                        varname, op, varval = condition.split('_')
                        varval = eval(varval)
                        if varname not in var_conditions:
                            var_conditions[varname] = defaultdict(None)

                        if op in ['∈', '∉']:
                            values = var_conditions[varname].get(op, set())
                            var_conditions[varname][op] = values | set(varval)
                        elif op == '≥':
                            value = var_conditions[varname].get(op, float('+inf'))
                            var_conditions[varname][op] = min(value, varval)
                        elif op == '<':
                            value = var_conditions[varname].get(op, float('-inf'))
                            var_conditions[varname][op] = max(value, varval)

                    predicates = []
                    for varname, merged_conditions in var_conditions.items():
                        operators = merged_conditions.keys()
                        if set(['∈', '∉']) & operators:
                            #* Categorical var
                            assert not set(['<', '≥']) & operators, f"{varname} has mixed {merged_conditions=}"
                            _invals = merged_conditions.get('∈', set())
                            _outvals = merged_conditions.get('∉', set())
                
                            invals = _invals - _outvals
                            outvals = _outvals - _invals
                            domain = set(self.domains[varname])

                            #* Use the most succinct representation
                            if invals:
                                diffvals = domain - invals
                                if len(diffvals) < len(invals):
                                    outvals |= diffvals
                                    invals = set()
                            if outvals:
                                diffvals = domain - outvals
                                if len(diffvals) < len(outvals):
                                    invals |= diffvals
                                    outvals = set()

                            predicate = ''
                            for val in invals:
                                predicate += f"Eq({varname}, {val})|"
                            predicate = predicate[:-1]
                            if len(invals) > 1:
                                predicate = '( ' + predicate + ' )'
                            if predicate:
                                predicates.append(predicate)

                            for val in outvals:
                                predicates.append(f"Ne({varname}, {val})")
                        else:
                            assert set(['<', '≥']) & operators, f"{varname} has no recognizd {merged_conditions=}"
                            #TODO: Force integer typing.
                            valmin = merged_conditions.get('≥', float('-inf'))
                            valmax = merged_conditions.get('<', float('+inf'))
                            if valmin >= valmax:
                                #TODO: Accumulate logits to decide which condition to take. Discard all together for now.
                                print(f"[Conflicting condition!!!]: {varname=}:{merged_conditions}")
                            else:
                                if valmin > float('-inf'):
                                    if self.dtypes[varname] == 'int':
                                        valmin = int(valmin)
                                    predicates.append(f"({varname} > {valmin})")
                                if valmax < float('+inf'):
                                    if self.dtypes[varname] == 'int':
                                        valmax = int(valmax)
                                    predicates.append(f"({varname} <= {valmax})")
                    
                    if predicates:
                        premise = ' & '.join(predicates)
                        if targetvar in self.categoricals:
                            conclusion = f"Eq({targetvar}, {targetcls})"
                        else:
                            targetmin = record['target_range']['min']
                            targetmax = record['target_range']['max']
                            if self.dtypes[targetvar] == 'int':
                                targetmin, targetmax = int(targetmin), int(targetmax)
                            conclusion = f"(({targetvar}>={targetmin}) & ({targetvar}<={targetmax}))"
                        rule = f"({premise}) >> {conclusion}"
                        rules.add(rule)
            learned_rules |= rules
            log.info(f"Extracted {len(rules)} rules from trees of {target}.")
        log.info(f"Total rules extracted: {len(learned_rules)}")
        return learned_rules

    def extract_conditions_from_treepaths(self) -> Set[str]:
        #TODO: Move to config
        MIN_GAIN = 0
        MIN_LOGIT = 0
        #* {target: {label1: [conditions1, conditions2, ...], label2: [...]}}
        all_pathconditions = {}
        for target, all_treepaths in self.extract_paths_from_all_trees().items():
            all_pathconditions[target] = defaultdict(list)
            encoder = self.label_encoders.get(target, None)
            for treeidx, paths in enumerate(all_treepaths):
                xgbtree: XGBClassifier|XGBRegressor = self.trees[target][treeidx]
                n_classes = getattr(xgbtree, 'n_classes_', 1)
                useless_splits = XgboostTreeLearner.get_useless_splits(xgbtree)
                label_map = dict(zip(encoder.transform(encoder.classes_), encoder.classes_)) \
                    if encoder else {}
                
                leafranges = {}  
                if not encoder:
                    #* Compute leaf ranges for regression trees
                    features = xgbtree.get_booster().feature_names
                    X = self.examples.copy()
                    #! Deal with bug in XGBoost3.0.2 where categoricals are not handled correctly at inference
                    X[self.categoricals] = X[self.categoricals].astype(int)
                    X = X[features]
                    leaves = xgbtree.apply(X)
                    # Flatten since we have only one tree
                    leaf_indices = leaves.flatten()
                    # Combine with target values
                    df = pd.DataFrame({'leaf': leaf_indices, 'target': self.examples[target]})
                    # Group by leaf and get min/max
                    leafrangedf = df.groupby('leaf')['target'].agg(['min', 'max'])
                    leafrangedf.reset_index(inplace=True)
                    for i, row in leafrangedf.iterrows():
                        leafranges[int(row['leaf'])] = {
                            'min': row['min'],
                            'max': row['max'],
                        }

                #* Group paths by target class
                for path in paths:
                    if path['Logit'] <= MIN_LOGIT: continue

                    pathcondition = {
                        'pathid': path['LeafID'],
                        'logit': path['Logit'],
                        'target_range': None,  # For regression trees
                    }
                    conditions = []
                    for split in path['Path']:
                        if split['Gain'] <= MIN_GAIN: continue
                        if split['NodeID'] in useless_splits: continue

                        varname = split['Feature']
                        if split['Split'] is not None:
                            op = '<' if split['Direction']=='Yes' else '≥'
                            condition = f"{varname}_{op}_{split['Split']}"
                        else:
                            assert split['Category'] is not None
                            op = '∈' if split['Direction'] == 'Yes' else '∉'
                            categories = [
                                #* Index into the original category
                                self.examples[varname].astype('category').cat.categories[int(v)] 
                                for v in split['Category']
                            ]
                            condition = f"{varname}_{op}_{categories}"
                        conditions.append(condition)

                    if conditions:
                        pathcondition['conditions'] = conditions
                        if n_classes > 1:
                            target_cls = path['Tree'] % n_classes
                            label = label_map[target_cls]
                            all_pathconditions[target][label].append(pathcondition)
                        else:
                            #* Regression tree
                            leafid = int(path['LeafID'].split('-')[-1])
                            if leafid in leafranges:
                                pathcondition['target_range'] = leafranges[leafid]
                                all_pathconditions[target][leafid].append(pathcondition)
        return all_pathconditions
                        
    @staticmethod
    def get_useless_splits(model: XGBClassifier|XGBRegressor) -> Set[int]:
        """Find the split whose two leaves have the same logits."""
        tree_df = model.get_booster().trees_to_dataframe()
        useless_splits = []
        for _, node in tree_df[tree_df['Feature'] != 'Leaf'].iterrows():
            yes_leaf = tree_df[(tree_df['Tree'] == node['Tree']) & (tree_df['ID'] == node['Yes'])]
            no_leaf = tree_df[(tree_df['Tree'] == node['Tree']) & (tree_df['ID'] == node['No'])]
        
            if not yes_leaf.empty and not no_leaf.empty:
                if yes_leaf.iloc[0]['Gain'] == no_leaf.iloc[0]['Gain']:
                    useless_splits.append( node['ID'] )
                    # useless_splits.append((node['Tree'], node['ID'], node['Feature']))
        return useless_splits
    
    def extract_paths_from_all_trees(self):
        """Extract path conditions from all trees."""
        all_tree_paths = defaultdict(list)
        for target, trees in self.trees.items():
            for treeidx, model in enumerate(tqdm(
                trees, desc=f"Extracting paths from trees of {target}"
            )):
                paths = self.extract_tree_paths(model)
                #* Paths could be empty `{}`, but keep it 
                #*  to match the indexing of `self.trees` to `all_tree_paths`
                all_tree_paths[target].append(paths)

        #* {target: [{tree1_paths}, {tree2_paths}, ...]}
        return all_tree_paths
    
    def extract_tree_paths(self, model: XGBClassifier|XGBRegressor):
        """
        Extracts decision paths from the last N trees in an XGBoost model, including categorical splits.
        N is the number of classes for classifiers, and 1 for regressors.

        Parameters:
        - model: Trained XGBoost model (e.g., XGBClassifier or XGBRegressor)

        Returns:
        - List of dictionaries, each representing a decision path from root to leaf.
        """
        booster = model.get_booster()
        tree_df = booster.trees_to_dataframe()

        # Preprocess node mapping per tree
        tree_group = tree_df.groupby('Tree')
        id_to_row_by_tree = {
            tree_idx: group.set_index('ID').to_dict(orient='index')
            for tree_idx, group in tree_group
        }

        leaf_nodes = tree_df[tree_df['Feature'] == 'Leaf']
        paths = []

        for _, leaf in leaf_nodes.iterrows():
            tree_index = leaf['Tree']
            leaf_id = leaf['ID']
            leaf_value = leaf['Gain']  # Leaf value

            id_to_row = id_to_row_by_tree[tree_index]
            path = []
            current_id = leaf_id
            visited = set()

            while True:
                if current_id in visited:
                    print(f"⚠️ Infinite loop detected at node {current_id} in tree {tree_index}")
                    break
                visited.add(current_id)

                # Find parent node
                parent_id = next(
                    (pid for pid, row in id_to_row.items()
                    if row.get('Yes') == current_id or row.get('No') == current_id),
                    None
                )

                if parent_id is None:
                    break  # Reached root

                parent = id_to_row[parent_id]
                direction = 'Yes' if parent['Yes'] == current_id else 'No'

                condition = {
                    'Tree': tree_index,
                    'NodeID': parent_id,
                    'Feature': parent['Feature'],
                    'Split': parent['Split'] if not np.isnan(parent['Split']) else None,
                    'Gain': parent['Gain'],
                    'Category': parent.get('Category'),
                    'Direction': direction
                }
                path.append(condition)
                current_id = parent_id

            path.reverse()

            paths.append({
                'Tree': tree_index,
                'LeafID': leaf_id,
                'Logit': leaf_value,
                'Path': path
            })

        return paths
    
    