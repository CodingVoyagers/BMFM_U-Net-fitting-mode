
import pandas as pd
import joblib
from getAllRules import *
"""
Returns the number of rules contained in each tree: dict {0:8,1:8}
"""


def gainTreesSplit(gbdt_tree,allRules): #Return the per-tree rule cutoffs
    splitNode = {}
    for i in range(gbdt_tree.tree_generator.n_estimators_):
        nums = 0
        for id, r in enumerate(allRules):
            if (r.fromTree == i + 1):# i+1 because the tree starts at number 1
                nums += 1
        splitNode[i] = nums
    return splitNode

def saveSplit():
    path = './model/GBDT.pkl'
    train_data = pd.read_csv('./data/feature_train.csv')
    gbdt_tree = joblib.load(path)

    allRules = gbdt_tree.rule_ensemble.rulesAll
    treeSplitNode = gainTreesSplit(gbdt_tree, allRules)  # per-tree-rule-division dict
    path = './RULE/everyTree1.pkl'
    joblib.dump(treeSplitNode, path)

    train_x = train_data.drop("zLabel", axis=1)
    features = train_x.columns
    tree_list = [[x] for x in gbdt_tree.tree_generator.estimators_]

    rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                 result=None,
                                 label=0,
                                 feature_names=features)
    allRules = rule_ensemble.rulesAll
    treeSplitNode = gainTreesSplit(gbdt_tree, allRules)  # per-tree-rule-division dict
    path = './RULE/everyTree0.pkl'
    joblib.dump(treeSplitNode, path)

    rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                 result=None,
                                 label=2,
                                 feature_names=features)
    allRules = rule_ensemble.rulesAll
    treeSplitNode = gainTreesSplit(gbdt_tree, allRules)  # per-tree-rule-division dict
    path = './RULE/everyTree2.pkl'
    joblib.dump(treeSplitNode, path)

    rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                 result=None,
                                 label=3,
                                 feature_names=features)
    allRules = rule_ensemble.rulesAll
    treeSplitNode = gainTreesSplit(gbdt_tree, allRules)  # per-tree-rule-division dict
    path = './RULE/everyTree3.pkl'
    joblib.dump(treeSplitNode, path)


if __name__ == '__main__':
   saveSplit()