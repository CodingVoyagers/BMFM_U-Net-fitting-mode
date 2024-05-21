import joblib
import pandas as pd
import csv
from getAllRules import *
from  getTreeRulesNum import *

"""
Store rules in tree order csv file

Get the judgment rule for label 
"""
class Rules1Ext():
    def __init__(self, rule, ratio,fromTree):
        self.rule = rule
        self.ratio = ratio
        self.useRatio = 0
        self.fromTree = fromTree # This rule comes from the tree index
    def __str__(self):
        return self.rule  # Return rules here

    def __repr__(self):
        return self.__str__()

def rules_list(label):
    rules1 = pd.read_csv(r'./RULE/satisfy_rules'+str(label) +'.csv')
    rulesOthers = pd.read_csv(r'./RULE/satisfy_rulesExcept'+ str(label) +'.csv')

    data1 = rules1.drop("index", axis=1)
    features = data1.columns.values.tolist()
    data1 = data1.values
    data2 = rulesOthers.drop("index", axis=1).values
    path = './RULE/everyTree'+str(label) +'.pkl'
    treeSplitNode = joblib.load(path)
    key = 0
    count = 0
    rule = []  #Single Tree Rules[ ,  ,  ,  ]
    rules = [] #Rules for all trees [[], [], []]
    rules_temp = []
    for r in range(data1.shape[1]): # Iterate over all individual rules
        if(count == treeSplitNode[key]):
            if(len(rule) > 0):
                rules.append(rule)
            rule = []
            count = 0
            key += 1
        count += 1
        num1 = (data1[:,r].tolist()).count(1)
        num2 = (data2[:,r].tolist()).count(1)
        allPatchNum = num1 + num2
        if allPatchNum == 0:
            continue
        if (num1/allPatchNum - num2 /allPatchNum >0.4):
            ruleObj = Rules1Ext(features[r],num1/allPatchNum,key+1)
            rule.append(ruleObj)
            rules_temp.append(ruleObj)
    rules.append(rule)
    for i in range(len(rules_temp)-1):
        j = i+1
        while(j <= len(rules_temp)-1):
            if(rules_temp[i].ratio < rules_temp[j].ratio):
                temp = rules_temp[i]
                rules_temp[i] = rules_temp[j]
                rules_temp[j] = temp
            j = j+1
    path = './RULE/rules'+str(label) +'.pkl'
    joblib.dump(rules_temp,path)


def train_rules(label):

    train_data = pd.read_csv('./data/feature_train.csv')
    train_x = train_data.drop("zLabel", axis=1)
    features = train_x.columns
    path = './model/GBDT.pkl'
    gbdt_tree = joblib.load(path)

    tree_list = [[x] for x in gbdt_tree.tree_generator.estimators_]
    rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                 result=None,
                                 label=label,
                                 feature_names=features)
    allRules = rule_ensemble.rulesAll
    # Save all rules
    temp = allRules.copy()
    temp.insert(0, 'index')
    X1Rules = rule_ensemble.transform(gbdt_tree.predictTrue1)
    X1Index = gbdt_tree.flag_1  # Correctly predicted to be the index of sample 1

    X2Rules = rule_ensemble.transform(gbdt_tree.predictTrue2)
    X2Index = gbdt_tree.flag_2  # Correctly predicted to be the index of sample 2

    X3Rules = rule_ensemble.transform(gbdt_tree.predictTrue3)
    X3Index = gbdt_tree.flag_3  # Correctly predicted to be the index of sample 3

    X0Rules = rule_ensemble.transform(gbdt_tree.predictTrue0)
    X0Index = gbdt_tree.flag_0  # Correctly predicted to be the index of sample 0

    checkRules=[X0Rules,X1Rules,X2Rules,X3Rules]
    checkIndex=[X0Index,X1Index,X2Index,X3Index]


    with open(r'./RULE/satisfy_rules'+ str(label) +'.csv', "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(temp)
        for index, item in enumerate(checkRules[label]):
            value = item.tolist()
            value.insert(0, checkIndex[label][index])
            writer.writerow(value)
        file.close()
    with open(r'./RULE/satisfy_rulesExcept'+ str(label) +'.csv', "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(temp)
        for idx in len(checkRules):
            if(idx == label):
                continue
            for index, item in enumerate(checkRules[label]):
                value = item.tolist()
                value.insert(0, checkIndex[label][index])
                writer.writerow(value)
        file.close()

    print("Label"+ str(label) +" Sample Rule Extraction Complete !")
    num = []
    for i in range(gbdt_tree.tree_generator.n_estimators_):
        for id, r in enumerate(allRules):
            if (r.fromTree == i + 1):
                num.append(str(r))
    data = pd.read_csv(r'./RULE/satisfy_rules'+ str(label) +'.csv')
    index = data[data.columns[0]].values
    data = data.drop("index", axis=1)
    data = data[num]
    data.insert(0, 'index', index)
    data.to_csv(r'./RULE/satisfy_rules'+ str(label) +'.csv', index=None)

    data = pd.read_csv(r'./RULE/satisfy_rulesExcept'+ str(label) +'.csv')
    index = data[data.columns[0]].values
    data = data.drop("index", axis=1)
    data = data[num]
    data.insert(0, 'index', index)
    data.to_csv(r'./RULE/satisfy_rulesExcept'+ str(label) +'.csv', index=None)


if __name__ == '__main__':
    saveSplit() # Returns the number of rules contained in each tree: dict {0:8,1:8}
    label = 3
    train_rules(label)
    rules_list(label)