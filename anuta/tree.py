import math
import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.tree import H2OTree, H2OLeafNode, H2OSplitNode
from typing import *
import itertools
from collections import defaultdict
from time import perf_counter
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
        self.target_to_features = defaultdict(list)
        for target in variables:
            features = [v for v in variables if v != target]
            for n in range(1, len(features)+1):
                self.target_to_features[target] += itertools.combinations(features, n)
        
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
        total_trees = len(self.target_to_features) * \
            len(self.target_to_features[list(self.target_to_features)[0]])
        log.info(f"Learning {total_trees} trees from {len(self.examples)} examples.")
        
        start = perf_counter()
        treeid = 1
        for target, feature_group in self.target_to_features.items():
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
                print(f"... Trained {treeid}/{total_trees} trees ({target=}).", end='\r')
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
        Theory.save_constraints(sprules, f'treerules_{self.dataset}.pl')
        
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
            log.info(f"Extracted {len(learned_rules)} rules from trees of {target}.")
        log.info(f"Total rules extracted: {len(learned_rules)}")
        return learned_rules
    
    def extract_paths_from_all_trees(self):
        """Extract path conditions from all trees."""
        all_tree_paths = defaultdict(list)
        for target, trees in self.trees.items():
            for treeidx, dtree in enumerate(tqdm(
                trees, desc=f"Extracting paths from trees of {target}"
            )):
                # print(f"Features: {self.target_to_features[target][treeidx]}")
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
 
