import numpy as np
from collections import Counter
from random import shuffle
from itertools import chain

TARGET_ATTRIBUTE = 9


class DecisionTreeNode:
    def __init__(self, is_leaf=False, prediction=None, attribute=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.attribute = attribute
        self.children = {}

#calculate entropy of a given label set
def entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    return -sum((count/total_count) * np.log2(count/total_count) for count in label_counts.values())

#calculate information gain of a potential split
def information_gain(data, split_attribute,labels):
    #get total entropy
    total_entropy = entropy(labels)
    #get all values and the count for each
    values, counts = zip(*Counter([record[split_attribute] for record in data]).items())
    weights = [count/len(data) for count in counts]
    weighted_entropy = sum(weights[i] * entropy([label for record, label in zip(data,labels) if record[split_attribute] == v]) for i, v in enumerate(values))
    return total_entropy - weighted_entropy

def build_tree(data, attributes, labels, min_examples=1):
    #stop the recursion if we have less records than the min examples for training
    if len(data) <= min_examples or not attributes:
        return DecisionTreeNode(is_leaf=True, prediction=Counter(labels).most_common(1)[0][0]) #prediction is equal to the most common label
    else:
        #get all the labels from the data
        labels = [record[TARGET_ATTRIBUTE] for record in data]
        #get the information gain for all the attributes
        ig = {attr: information_gain(data, attr, labels) for attr in attributes}
        #select the attribute with the highest information gain
        max_ig_attr = max(ig, key=ig.get)
        #create a new decision node using the best attribute
        node = DecisionTreeNode(attribute=max_ig_attr)
        #remove the best attribute from the attribute list for the next iteration
        attributes = [attr for attr in attributes if attr != max_ig_attr]
        #create a new decision node for each unique value of the best attribute
        for attr_val in set(record[max_ig_attr] for record in data):
            #filter only the data which is equal to the attribute value
            child_data = [record for record in data if record[max_ig_attr] == attr_val]
            if not child_data:
                #if the split is empty, create a leaf node with the most common label
                prediction = Counter(labels).most_common(1)[0][0]
                child_node = DecisionTreeNode(is_leaf=True, prediction=prediction)
            else:
                child_node = build_tree(child_data, attributes, labels, min_examples)
            #add the child node to the current node
            node.children[attr_val] = child_node
        return node

def k_fold_split(data, k):
    #shuffle data
    shuffle(data)
    fold_size = len(data) // k
    return [data[i*fold_size:(i+1)*fold_size] for i in range(k)]

def predict(tree, prediction):
    while not tree.is_leaf:
        try:
            tree = tree.children[prediction[tree.attribute]]
        except KeyError:
            return None
    return  tree.prediction

def evaluate(tree, test_data):
    correct = 0
    for prediction in test_data:
        if predict(tree,prediction) == prediction[TARGET_ATTRIBUTE]:
            correct +=1
    return correct / len(test_data)

# fetch dataset 
breast_cancer_data = []
data_set_path = ''
with open(data_set_path, 'r') as file:
    breast_cancer_data = file.readlines()

parsed_data = [line.strip().split(',') for line in breast_cancer_data]

# attributes will hold all of the available attributes
attributes = list(range(len(parsed_data[0])))
#dict_data will hold each line of our data in a map 
#where the key is the number of the attribute and the value is the attribute
dict_data = [{attr: value for attr, value in zip(attributes, record)} for record in parsed_data]
#remove the last attribute from the list because it is the target
attributes.pop()

#default labels
labels = [x[TARGET_ATTRIBUTE] for x in parsed_data]
#min examples for training
min_examples = 5  

accuracies = []
kfold_splits = 5

#split the data into folds
folds = k_fold_split(dict_data,kfold_splits)
for i in range(kfold_splits):
    #construct the train set
    train_set = list(chain.from_iterable(fold for index,fold in enumerate(folds) if index != i))
    test_set = folds[i]
    #build te decision tree
    tree = build_tree(train_set, attributes, labels, min_examples)
    accuracy = evaluate(tree, test_set)
    accuracies.append(accuracy)

print(sum(accuracies)/kfold_splits)