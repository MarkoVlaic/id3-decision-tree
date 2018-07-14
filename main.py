import pandas as pd
from math import log2

class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []
        self.transitions = []
        self.outcome_node = False

    def add_child(self, child, transition):
        self.children.append(child)
        self.transitions.append(transition)

class DecisionTree(object):
    def __init__(self, dataset=None, label_header=None):
        if dataset is None or label_header is None:
            raise Exception('dataset and label_header arguments must be specified')

        self.dataset = dataset
        self.label_header = label_header
        self.root = None

    def __p(self, c, S):
        return list(S).count(c)/len(S)

    def __H(self, S):
        res = 0

        for i in set(S):
            y = self.__p(i, S)
            res += -y*log2(y)

        return res

    def __divide_by_category(self, category, S):
        d = {}
        #Data labels (yes/no) are saved into d as d[type] = [labels from type] where type is element of category
        #Explanation: category can be outlook, type = {sunny, rainy, overcast} so the result may be d = {'sunny': ['yes', 'no'], 'rainy': ['no', no], 'overcast': ['yes', 'yes', 'no']}

        #print('Divide by category set', S)

        for i in range(len(S[category])):
            elem = S[category][i]
            label = S[self.label_header][i]
            if elem in d:
                d[elem].append(label)
            else:
                d[elem] = [label]
        return d

    def __entropy_for_category(self, category, S):
        d = self.__divide_by_category(category, S)
        res = 0

        for i in set(S[category]):
            res += len(d[i])/len(S[category]) * self.__H(d[i])

        return res

    def __gen_set_for_category(self, category, S):
        #print('Category',category)
        res = {}

        for key in set(S[category]):
            res[key] = {}
            for key2 in S:
                res[key][key2] = []

        for key in set(S[category]):
            for i in range(len(S[category])):
                if S[category][i] == key:
                    for key2 in S:
                        res[key][key2].append(S[key2][i])

        return res


    def train(self, S=None):
        if S is None:
            S=self.dataset

        prev_entropy= self.__H(S[self.label_header])

        if prev_entropy == 0:
            node = Node(S[self.label_header][0])
            node.outcome_node = True
            return node

        max_gain = 0
        ent = 0
        max_cat = ''

        for category in S:
            #print('Category {} entropy {}'.format(category, self.__entropy_for_category(category, S)))
            if category == self.label_header:
                continue

            cat_ent = self.__entropy_for_category(category, S)
            gain = prev_entropy - cat_ent

            if gain > max_gain:
                max_gain = gain
                ent = cat_ent
                max_cat = category


        print('Category {} entropy {} gain {}'.format(max_cat, ent, max_gain))

        node = Node(max_cat)

        if self.root is None:
            self.root = node

        #d = self.__divide_by_category(max_cat, S)
        d = self.__gen_set_for_category(max_cat, S)

        for key in d:
            if self.__H(d[key]) != 0:
                print('transition', key)
                node.add_child(self.train(S=d[key]), transition=key)
            else:
                node.add_child(d[key][0], transition=key)
                node.outcome_node = True

        return node

    def classify(self,data):
        node = self.root

        while not node.outcome_node:
            #print('node', node.value)
            for i in range(len(node.transitions)):
                #print('transition {} data {}'.format(node.transitions[i], data[node.value]))
                if(node.transitions[i] == data[node.value]):
                    #print('Transition', node.transitions[i])
                    node = node.children[i]
                    #print('next node', node.value)
                    break
        return node.value

    def print_out(self, node=None):
        print('-'*15)
        if node is None:
            node = self.root

        print('Value', node.value)

        for i in range(len(node.children)):
            print('Child {} transition {}'.format(node.children[i].value, node.transitions[i]))

        for child in node.children:
            self.print_out(child)

if __name__ == '__main__':
    data = pd.read_csv('dataset.csv')
    dt = DecisionTree(dataset=data, label_header='play')
    dt.train()
    print('Training complete')

    unknown = pd.read_csv('unknown.csv')
    headers = list(unknown)

    for i in range(unknown.shape[0]):
        #print(i, headers[0], headers[-1])
        print(dt.classify(unknown.loc[i, headers[0]:headers[-1]]))
