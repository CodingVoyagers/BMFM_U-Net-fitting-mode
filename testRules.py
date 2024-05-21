import pandas as pd
import joblib
import time
from rulesExtraction import Rules1Ext
from sklearn.metrics import classification_report
"""
Test extraction rules label
"""

def test_labelRules(test):
    path = './RULE/rules' + str(test) +'.pkl'
    rules = joblib.load(path)
    data = pd.read_csv(r'./data/feature_test.csv')
    allColumns = (data.columns.values).tolist()[1:]
    ################################
    label = data.zLabel.values.tolist()
    for index,val in enumerate(label):
        if val != test:            label[index] = 0
        else:
            continue
    data = data[data.columns[1:]].values
    data = data.tolist()
    predict = []
    count = 0
    time1 = time.time()
    for samples in data:
        count += 1
        print("Sample %s(%s) in judgment..." % (count, len(data)))
        temp = 0
        for rule in rules:
            flag = 0  # Whether to end the sample judgment flag
            ruleList = str(rule).split(' & ')
            for node in ruleList:
                feature = node.split(" ")[0]  # Features to be compared
                op = node.split(" ")[1]  # 判Judgment symbol
                judgeValue = float(node.split(" ")[2])  # Judgment value
                sampleValue = float(samples[allColumns.index(feature)])  # To need to compare the features
                if (op == "<="):
                    if (sampleValue > judgeValue):
                        flag = 1  # The sample does not comply with this rule
                        break
                if (op == ">"):
                    if (sampleValue <= judgeValue):
                        flag = 1  # The sample does not comply with this rule
                        break
            if (flag == 0):  # After a rule, if all the rules are satisfied, the rule traversal judgment is jumped
                if(temp == 0):
                    predict.append(test)
                    print("This sample satisfies some of the rules for extraction...")
                    temp = 1
                rule.useRatio += 1

        if (temp == 0):  # All rules do not satisfy this sample, and the predicted value is 0
            predict.append(0)
    time2 = time.time()
    print("\nThe time required to judge the sample is: %s s"%(time2-time1))
    t = classification_report(label,predict, target_names=['0', str(test)],digits=4)
    print(t)



if __name__ == '__main__':
    label = 3
    test_labelRules(label)  #Test extraction rules label