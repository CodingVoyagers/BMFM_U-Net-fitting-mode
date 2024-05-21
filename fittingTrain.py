import time
import pandas as pd
import joblib
from Gbdt import Tree
from sklearn.metrics import accuracy_score

def fit_unet_GDBT():

    train_data = pd.read_csv('./data/feature_train.csv')
    train_y = train_data.zLabel.values
    train_x = train_data.drop("zLabel", axis=1)
    features = train_x.columns
    train_x = train_x.values
    gbdt_tree = Tree()
    time1 = time.time()
    print("Start training...")
    gbdt_tree.fit(train_x, train_y, feature_names=features)
    time2 = time.time()

    print("\nThe training time is： %s s"%(time2-time1))

    """
    Save training model
    """
    path = 'model/GBDT.pkl'
    joblib.dump(gbdt_tree,path)
    predict = gbdt_tree.predict(train_x)
    accuracy = accuracy_score(predict, train_y)
    print('Train Accuracy:', accuracy)

if __name__ == '__main__':
    fit_unet_GDBT()