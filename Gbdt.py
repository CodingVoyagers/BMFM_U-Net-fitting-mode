import time
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from functools import reduce
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score


class RuleCondition():
    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 count,
                 feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name
        self.count = count

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        if self.operator == "<=":
            res = 1 * (X[:, self.feature_index] <= self.threshold)  # The res value is 1 if it is satisfied, and 0 if it is not.
        elif self.operator == ">":
            res = 1 * (X[:, self.feature_index] > self.threshold)  # The res value is 1 if it is satisfied, and 0 if it is not.
        return res

    # Determine if two objects are equal
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class Rule():
    def __init__(self,
                 rule_conditions, prediction_value, fromTree, friedman_mse, samples):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])  # Percentage of samples reaching this rule
        self.prediction_value = prediction_value
        self.rule_direction = None
        self.fromTree = fromTree
        self.friedman_mse = friedman_mse
        self.samples = samples

    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x, y: x * y, rule_applies)

    def __str__(self):
        return " & ".join([x.__str__() for x in self.conditions])  # 这里返回规则

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def extract_rules_from_tree(tree, count, feature_names=None):
    """
    count Records the tree to which the extraction rule belongs
    """
    rules = set()

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        # rules
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           count=count,
                                           feature_name=feature_name
                                           )
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []
            ## if not terminal node
            """
            1, for the node array left and right, -1 represents the leaf node
            2, for the threshold threshold, represents the current node selection of the corresponding characteristics of 
            the split threshold, generally ≤ the threshold into the left child node, otherwise into the right child node, 
            -2 represents the leaf node characteristics of the threshold value
            3, the value on the graph is a floating point number with 3 decimal places.
            4, left array is the root node (excluding the root node itself) left subtree traversal of 
            the sequence of node ids in the preorder (including leaf nodes)
            5, right array is the root node (excluding the root node itself) of the right subtree of 
            the node id sequence of the forward traversal (including leaf nodes)
            6, threshold and features of the order is left --> right order
            7, features: size type, represents the current node used to split the feature index, 
            that is, in the training set with the first column of features for splitting
            8、n_node_samples：size type, represents the total number of samples falling into the node during training. 
            Obviously, the n_node_samples of the parent node will be equal to the sum of the n_node_samples of its left and right child nodes
            """
        if tree.children_left[node_id] != tree.children_right[node_id]:
            # If it's a leaf node left or right child node value is -1
            feature = tree.feature[node_id]# Current node feature number
            threshold = tree.threshold[node_id]# Current node feature threshold
            # It's also a priori traversal of the nodes
            left_node_id = tree.children_left[node_id]#Current node's left child
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)
            new_conditions = new_conditions
            right_node_id = tree.children_right[node_id]#Current node's right child
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else:  # a leaf node
            if len(new_conditions) > 0:
                # use & combine rules
                new_rule = Rule(new_conditions, tree.value[node_id][0][0], count, tree.impurity[node_id],
                                tree.n_node_samples[
                                    node_id]) # Instantiated rules include conditions (attributes in the RuleCondition class) prediction_value support Decision list paths
                                            # for each tree added to the Rule class
                rules.update([new_rule]) # Each decision path of all paths is an instantiated object of the Rule class
            else:
                pass  # tree only has a root node!
            return None

    traverse_nodes()

    return rules


class RuleEnsemble():
    def __init__(self,
                 tree_list,
                 result,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        """
         The reason it's a set is because of the rule to remove duplicates, 
         and set() in turn automatically removes duplicate elements
        """
        self.rulesAll = set()  # all rules set
        self.result = result  # gdbt prediction
        self._extract_rules()  # extract rules
        self.rulesAll = list(self.rulesAll)
        # self.rules_1 = list(self.rules_1)
        # self.rules_others = list(self.rules_others)

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble
        """
        count = 0
        for tree in self.tree_list:
            count += 1
            """
            1, for the node array left and right, -1 represents the leaf node
            2, for the threshold threshold, represents the current node selection of the corresponding characteristics of 
            the split threshold, generally ≤ the threshold into the left child node, otherwise into the right child node, 
            -2 represents the leaf node characteristics of the threshold value
            3, the value on the graph is a floating point number with 3 decimal places.
            4, left array is the root node (excluding the root node itself) left subtree traversal of 
            the sequence of node ids in the preorder (including leaf nodes)
            5, right array is the root node (excluding the root node itself) of the right subtree of 
            the node id sequence of the forward traversal (including leaf nodes)
            6, threshold and features of the order is left --> right order
            7, features: size type, represents the current node used to split the feature index, 
            that is, in the training set with the first column of features for splitting
            8、n_node_samples：size type, represents the total number of samples falling into the node during training. 
            Obviously, the n_node_samples of the parent node will be equal to the sum of the n_node_samples of its left and right child nodes.
            Here the rules that should be extracted to satisfy the tree of the predicted kind of this variable
            """
            rulesFromTree = extract_rules_from_tree(tree[0][1].tree_, count, feature_names=self.feature_names)
            self.rulesAll.update(rulesFromTree)
            print("All judgment rules for a category in the %sth tree have been extracted...."%count)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X1):
        """Transform dataset.
        Parameters
        ----------
        X:      array-like matrix, shape=(n_samples, n_features)
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list = list(self.rulesAll)
        # Here rule.transform calls the transform in the Rule() class
        # Perform transform() for each complete decision path.
        # Find the samples that satisfy a complete decision path A value of 1 satisfies the decision path.
        # T get behavioral samples after the column for the decision path,
        # each line corresponding to the column of 1 for the sample to meet the decision path of
        # the column of the rule (the final decision results are not right to say)
        print("\nRules being extracted that are correctly predicted as a class of labels...")
        rule1 = np.array([rule.transform(X1) for rule in rule_list]).T
        return rule1

    # def __str__(self):
    #     return (map(lambda x: x.__str__(), self.rules)).__str__()


class Tree(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            tree_size=6,
            sample_fract='default',
            max_rules=2000,
            memory_par=0.1,
            rand_tree_size=True,  # Whether to randomize leaf node size
            random_state=None):
        self.rand_tree_size = rand_tree_size
        self.sample_fract = sample_fract
        self.max_rules = max_rules
        self.memory_par = memory_par
        self.tree_size = tree_size
        self.random_state = random_state

    def fit(self, X, y, feature_names=None):
        """Fit and estimate linear combination of rule ensemble
        """
        ## Enumerate features if feature names not provided
        N = X.shape[0]  # the number of data
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names = feature_names  # features name

        n_estimators_default = int(np.ceil(self.max_rules / self.tree_size))
        self.sample_fract_ = min(0.5, (100 + 6 * np.sqrt(N)) / N)  #  N is the number of data

        self.tree_generator = GradientBoostingClassifier(#n_estimators=n_estimators_default,
        #                                                  n_estimators=num_estimators,
                                                         # max_leaf_nodes=self.tree_size,
                                                         learning_rate=self.memory_par,
                                                         subsample=0.7)
                                                         #random_state=self.random_state)#, max_depth=10)

        ## fit tree generator
        if not self.rand_tree_size:  # simply fit with constant tree size
            self.tree_generator.fit(X, y)
        else:  # random tree size
            np.random.seed(self.random_state)
            tree_sizes = np.random.exponential(scale=self.tree_size - 2,
                                               size=int(np.ceil(self.max_rules * 2 / self.tree_size)))
            tree_sizes = np.asarray([2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))], dtype=int)
            i = int(len(tree_sizes) / 4)
            while np.sum(tree_sizes[0:i]) < self.max_rules:
                i = i + 1
            tree_sizes = tree_sizes[0:i]
            self.tree_generator.set_params(warm_start=True)
            curr_est_ = 0
            for i_size in np.arange(len(tree_sizes)):
            # for i_size in np.arange(num_estimators):
                size = tree_sizes[i_size]
                time1 = time.time()
                self.tree_generator.set_params(n_estimators=curr_est_ + 1)
                self.tree_generator.set_params(max_leaf_nodes=size)
                random_state_add = self.random_state if self.random_state else 0
                self.tree_generator.set_params(
                    random_state=i_size + random_state_add)
                self.tree_generator.get_params()['n_estimators']
                self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))

                curr_est_ = curr_est_ + 1
                time2 = time.time()
                print("The %s/%s decision tree training is complete... ----- max_leaf_nodes %s individual ----- Time spent generating tree %s s" % (curr_est_,len(tree_sizes),size,time2-time1))
            self.tree_generator.set_params(warm_start=False)

        # tree_list = self.tree_generator.estimators_  # Visiting a tree
        tree_list = [[x] for x in self.tree_generator.estimators_]
        predictTrain = self.tree_generator.predict(X).tolist()  # Calculate the predictions on the training set and use them to extract the corresponding rules.
        self.train_accuracy = accuracy_score(y, predictTrain)
        self.flag_0 = []
        self.flag_1 = []
        self.flag_2 = []
        self.flag_3 = []
        id = -1
        for GT in y:
            # Create indexes for data samples with correct predictions
            id += 1  #  Indexing of predicted values
            if predictTrain[id] == GT:  # Predicted value is correct
                if GT == 1:
                    self.flag_1.append(id)
                elif GT == 2:
                    self.flag_2.append(id)
                elif GT == 3:
                    self.flag_3.append(id)
                else:
                    self.flag_0.append(id)
            else:
                continue
        # Use the indexes of the four labels created in the previous step to construct data samples that predict the correct four labels.
        self.predictTrue1 = np.zeros([len(self.flag_1), 444], dtype=np.float32)
        self.predictTrue2 = np.zeros([len(self.flag_2), 444], dtype=np.float32)
        self.predictTrue3 = np.zeros([len(self.flag_3), 444], dtype=np.float32)
        self.predictTrue0 = np.zeros([len(self.flag_0), 444], dtype=np.float32)
        # self.predictTrue1 = np.zeros([len(self.flag_1), X.shape[1]], dtype=np.float32)  #降维后可能特征维数小于444

        """
       Get the sample data correctly predicted for each label
        """
        count = -1  # count numbers
        for train_id in self.flag_1:
            # Construct 1 label samples
            count += 1
            self.predictTrue1[count, :] = X[train_id, :]

        count = -1   # count numbers
        for train_id in self.flag_2:
            # Construct 2 label samples
            count += 1
            self.predictTrue2[count, :] = X[train_id, :]

        count = -1   # count numbers
        for train_id in self.flag_3:
            # Construct 3 label samples
            count += 1
            self.predictTrue3[count, :] = X[train_id, :]

        count = -1   # count numbers
        for train_id in self.flag_0:
            # Construct 0 label samples
            count += 1
            self.predictTrue0[count, :] = X[train_id, :]


        """
        Here the dataset is constructed for rule extraction: 
        the dataset from which the rules are extracted is the training set and is correctly predicted
        """
        ## extract rules
        self.rule_ensemble = RuleEnsemble(tree_list=tree_list,
                                          result=predictTrain,
                                          feature_names=self.feature_names)
        """
        ## concatenate original features and rules
        # rule_ensemble Called transform(X) in the RuleEnsemble() class.
        # Xi_rules : Samples whose behavior correctly predicts label i Columns are decision paths, 
        and the corresponding column 1 in each row is the rule that the sample satisfies the decision path in that column 
        (whether the final decision is correct or not is another matter).
        """
        self.X1_rules = self.rule_ensemble.transform(self.predictTrue1)
        return self

    def predict(self, X):
        """Predict outcome for X
        """
        return self.tree_generator.predict(X)
