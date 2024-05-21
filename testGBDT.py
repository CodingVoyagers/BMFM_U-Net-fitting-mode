import pandas as pd
import joblib
from Gbdt import Tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
if __name__ == '__main__':
    """
    Loading training model
    """
    gbdt = joblib.load('./model/GBDT.pkl')

    test_data = pd.read_csv('./data/feature_test.csv')
    test_y = test_data.zLabel.values
    test_x = test_data.drop("zLabel", axis=1)
    test_x = test_x.values
    predict = gbdt.predict(test_x)
    accuracy = accuracy_score(predict, test_y)
    print('Accuracy:', accuracy)

    t = classification_report(test_y,predict, target_names=['0','1','2','3'])
    print(t)