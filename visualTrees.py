import pydotplus
from sklearn.tree import export_graphviz
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def drawTrees(tree_list,label,feature_names):

    """
    Visualize every tree
    """
    N = 0
    for dt in tree_list:
        N += 1
        dot_data = export_graphviz(dt[0][label], out_file=None,  # Save Label 1 Tree
                                   feature_names=feature_names,
                                   filled=True, rounded=True,
                                   class_names=[0, 1, 2, 3])
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('./tree_pic/label'+str(label)+'/' + str(N) + "_DTtree.png")
        print("The %sth tree of the label %s has been visualized!" %(label,N))

if __name__ == '__main__':
    """
    Load training model
    """
    path = './model/GBDT.pkl'
    gbdt_tree = joblib.load(path)
    tree_list = [[x] for x in gbdt_tree.tree_generator.estimators_]

    """
    Get feature_names
    """
    data = pd.read_csv('./data/feature_test.csv')
    data = data.drop("zLabel", axis=1)
    features = data.columns
    # Visualizing the number of judgment labels 1
    drawTrees(tree_list,1,features)