
import numpy as np
from functools import reduce

"""
Extract all rules for each dichotomy
"""


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
            res = 1 * (X[:, self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:, self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class Rule():
    def __init__(self,
                 rule_conditions, prediction_value, fromTree, friedman_mse, samples):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
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
        return " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def extract_rules_from_tree(tree, count, feature_names=None):
    rules = set()

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
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

        if tree.children_left[node_id] != tree.children_right[node_id]:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)
            new_conditions = new_conditions
            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else:  # a leaf node
            if len(new_conditions) > 0:

                new_rule = Rule(new_conditions, tree.value[node_id][0][0], count, tree.impurity[node_id],
                                tree.n_node_samples[
                                    node_id])
                rules.update([new_rule])
            else:
                pass  # tree only has a root node!
            return None

    traverse_nodes()

    return rules


class RuleEnsemble():
    def __init__(self,
                 tree_list,
                 result,
                 label,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.label = label
        self.rulesAll = set()
        self.result = result
        self._extract_rules()
        self.rulesAll = list(self.rulesAll)


    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble
        """
        count = 0
        for tree in self.tree_list:
            count += 1
            rulesFromTree = extract_rules_from_tree(tree[0][self.label].tree_, count, feature_names=self.feature_names)
            self.rulesAll.update(rulesFromTree)
            print("All judgment rules for a category in the %sth tree have been extracted...."%count)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X):
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
        print("\nRules being extracted that are correctly predicted as a class of labels...")
        rule1 = np.array([rule.transform(X) for rule in rule_list]).T
        return rule1

