import abc
import copy
import graphviz
import math
from ml.Model import Model
import pandas as pd
from typing import Tuple, Any
import uuid


class DecisionTreeClassifier(Model, abc.ABC):
    class Label:
        def __init__(self, name: str, values: list):
            self.name = name
            self.values = values

        def __repr__(self):
            return self.name + ': ' + str(self.values)

    class Node:
        def __init__(self, label: str = None):
            self.attr = None
            self.branches = {}
            self.info = []
            self.info_gain = -1
            self.label = label
            self.threshold = None

        def __getitem__(self, item):
            return self.__dict__[item]

        def __setitem__(self, key, value):
            self.__dict__[key] = value

        def __repr__(self):
            return str(self)

        def __str__(self):
            if self.label:
                return f'{self.label}'

            pretty_str = f'{self.attr:<8} = {self.info_gain:.3f}\n'
            pretty_str += f'{self.attr:<8} < {self.threshold}\n' if self.threshold else ''
            pretty_str += '-'*18 + '\n'
            for info_gain, attr in self.info:
                pretty_str += f'{attr:<8} = {info_gain:.3f}\n'

            return pretty_str

    def __init__(self, numerical_attributes: list = None, max_depth=2**63, label: Label = None):
        self._depth = 1
        self._graph = None
        self._heuristic = None
        self._max_depth = max_depth
        self._numerical_attributes = numerical_attributes
        self._label = label
        self._root = None

    def fit(self, df: pd.DataFrame, numerical_attributes: list = None, heuristic='entropy', max_depth=2**63):
        """
        :param df: dataframe needs to have column names and last column should be the label
        :param numerical_attributes: list of attributes (str) which have numerical value
        :param heuristic: [default = 'entropy', 'majority_error', 'gini_index']
        :param max_depth: max permitted depth of decision tree
        :return:
        """

        self.__init__(numerical_attributes, max_depth, self.Label(df.columns[-1], df[df.columns[-1]].tolist()))

        if heuristic == 'entropy':
            self._heuristic = self._entropy
        elif heuristic == 'majority_error':
            self._heuristic = self._majority_error
        elif heuristic == 'gini_index':
            self._heuristic = self._gini_index
        else:
            raise TypeError('heuristic')

        attributes = {}
        for attr in df.columns[:-1]:
            if attr in self._numerical_attributes:
                attributes[attr] = []
            else:
                attributes[attr] = list(set(df[attr]))

        row_filter = pd.Series([True] * len(df))

        self._root = self._id3(df, row_filter, attributes, 1)

    def predict(self, input_features: pd.Series):
        def dfs(node: DecisionTreeClassifier.Node, inp: pd.Series):
            if node.label:
                return node.label

            if node.attr in self._numerical_attributes:
                return dfs(node.branches[inp[node.attr] < node.threshold], inp)

            return dfs(node.branches[inp[node.attr]], inp)

        return dfs(self._root, input_features)

    def predict_batch(self, batch_input_features: pd.DataFrame) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for row in range(len(batch_input_features)):
            predictions[row] = self.predict(batch_input_features.iloc[row, :])

        return predictions

    def gen_tree(self) -> graphviz.Digraph:
        if self._graph:
            return self._graph

        self._graph = graphviz.Digraph()

        def dfs(node_: DecisionTreeClassifier.Node, node_id=str(uuid.uuid1())):
            self._graph.node(node_id, str(node_))
            for attr_val, adj_node in node_.branches.items():
                adj_node_id = str(uuid.uuid1())
                self._graph.node(adj_node_id, str(adj_node))
                self._graph.edge(node_id, adj_node_id, label=str(attr_val))
                dfs(adj_node, adj_node_id)

        dfs(self._root)
        return self._graph

    def view_tree(self):
        if not self._graph:
            self.gen_tree()

        self._graph.render()

    def _id3(self, df: pd.DataFrame, row_filter: pd.Series, attributes: dict, depth: int):
        self._depth = max(self._depth, depth)
        node = self.Node()

        if depth == self._max_depth:
            node.label = df.loc[row_filter].groupby(self._label.name).size().idxmax()
            return node

        if len(df.loc[row_filter, :].groupby(self._label.name).size()) == 1:
            node.label = df.loc[row_filter, self._label.name].values[0]
            return node

        for attr in attributes.keys():
            curr_info_gain, threshold = self._info_gain(df, row_filter, attributes, attr)
            node.info.append((curr_info_gain, attr))
            if node.info_gain < curr_info_gain:
                node.info_gain = curr_info_gain
                node.attr = attr
                node.threshold = threshold

        node.info.sort(reverse=True)
        new_attributes = copy.deepcopy(attributes)
        attr_values = new_attributes.pop(node.attr, [])

        if node.attr in self._numerical_attributes:
            new_row_filter = row_filter & (df[node.attr] < node.threshold)
            if new_row_filter.sum():
                node.branches[True] = self._id3(df, new_row_filter, new_attributes, depth + 1)
            else:
                node.branches[True] = self.Node(label=df.loc[row_filter, self._label.name].values[0])

            new_row_filter = row_filter & (df[node.attr] >= node.threshold)
            if new_row_filter.sum():
                node.branches[False] = self._id3(df, new_row_filter, new_attributes, depth + 1)
            else:
                node.branches[False] = self.Node(label=df.loc[row_filter, self._label.name].values[0])

            return node

        for val in attr_values:
            new_row_filter = row_filter & (df[node.attr] == val)
            if sum(new_row_filter) == 0:
                node.branches[val] = self.Node(df[row_filter].groupby(self._label.name).size().idxmax())
            else:
                node.branches[val] = self._id3(df, new_row_filter, new_attributes, depth + 1)

        return node

    @staticmethod
    def _entropy_log2(val):
        return math.log2(val) if val else 0

    def _entropy(self, ps):
        return -sum(p * self._entropy_log2(p) for p in ps)

    @staticmethod
    def _majority_error(ps) -> float:
        if not ps.empty:
            return 1 - max(ps)

        return 0

    @staticmethod
    def _gini_index(ps):
        return 1 - sum(p * p for p in ps)

    def _info_gain(self, df: pd.DataFrame, row_filter: pd.Series, attributes: dict, attr: str) -> Tuple[float, Any]:
        if attr in self._numerical_attributes:
            return self._info_gain_numerical(df, row_filter, attr)

        s_len = row_filter.sum()
        gain = self._heuristic((df.loc[row_filter, :].groupby(self._label.name).size()) / s_len)
        for attr_val in attributes[attr]:
            freq = df.loc[row_filter & (df[attr] == attr_val), :].groupby(self._label.name).size()
            sv_len = freq.sum()
            gain -= ((sv_len / s_len) * self._heuristic(freq / sv_len))

        return gain, None

    def _info_gain_numerical(self, df: pd.DataFrame, row_filter: pd.Series, attr: str) -> Tuple[float, float]:
        s_len = row_filter.sum()
        gain = self._heuristic((df.loc[row_filter, :].groupby(self._label.name).size()) / s_len)

        threshold = df.loc[row_filter, attr].median()  # FIXME: use function agg

        freq = df.loc[row_filter & (df[attr] < threshold), :].groupby(self._label.name).size()
        sv_len = freq.sum()
        gain -= ((sv_len / s_len) * self._heuristic(freq / sv_len))

        freq = df.loc[row_filter & (df[attr] >= threshold), :].groupby(self._label.name).size()
        sv_len = freq.sum()
        gain -= ((sv_len / s_len) * self._heuristic(freq / sv_len))

        return gain, threshold
