# Program to classify leaf data using random forests.

# Package imports:
import numpy as np
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Main function:

# Control variables:
random_seed = 193777
test_size = 0.15

print("General info:")
print(">> Test size: %.2f%%" % (test_size*100))
print(">> Seed used: %d" % (random_seed))
print("")

# Data extraction:
leaf_dataset = pandas.read_csv('data/leaf.csv', header = None)
leaf_features = leaf_dataset.values[:,2:]
leaf_labels = list(map(int, leaf_dataset.values[:,0]))

# Train and test splitting:
train_features, test_features, train_labels, test_labels = \
model_selection.train_test_split(leaf_features,
                                 leaf_labels,
                                 test_size = test_size,
                                 random_state = random_seed)

# Initializing classifiers:
classifiers = []
classifiers.append(('Default RF',
                    RandomForestClassifier(n_estimators = 100,
                                           criterion = 'gini',
                                           max_depth = None,
                                           min_samples_split = 2,
                                           min_samples_leaf = 1,
                                           max_features = 'auto',
                                           max_leaf_nodes = None)))
