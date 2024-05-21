"""
Divide the data training set, test set
"""
import pandas as pd
from sklearn.model_selection import train_test_split
def splitData():
    data = pd.read_csv("./data/data.csv")
    target = 'zLabel'
    predictors = [x for x in data.columns if x not in [target]]
    train = data[predictors]
    label = data[target]
    train_x, test_x, train_y, test_y = train_test_split(train, label, test_size=0.2,random_state=41)
    train_y = train_y.values
    test_y = test_y.values
    train_x.insert(0,'zLabel',train_y)
    test_x.insert(0, 'zLabel', test_y)
    train_x.to_csv('./data/feature_train.csv', index=None)
    test_x.to_csv('./data/feature_test.csv', index=None)

if __name__ == '__main__':
    splitData()