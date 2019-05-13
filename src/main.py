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
scoring = 'accuracy'
test_size = 0.15

print("General info:")
print(">> Test size: %.2f%%" % (test_size*100))
print(">> Scoring type: ", scoring.capitalize())
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
classifiers.append(('Entropy-500 RF',
                    RandomForestClassifier(n_estimators = 500,
                                           criterion = 'entropy',
                                           max_depth = None,
                                           min_samples_split = 2,
                                           min_samples_leaf = 1,
                                           max_features = 'auto',
                                           max_leaf_nodes = None)))

# Cross-validation predictions and test results:
for name, rf in classifiers:
    print("Classifier: %s" % (name))

    # 10 fold cross-validation estimate:
    kfold = model_selection.KFold(n_splits = 10, random_state = random_seed)
    cv_results = model_selection.cross_val_score(rf, train_features,
                                                 train_labels, cv = kfold,
                                                 scoring = scoring)
    print(">> Cross-validation score: %f (%f)" %
          (cv_results.mean(), cv_results.std()))

    # Test results:
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    print(">> Accuracy: %f\n" % (accuracy_score(test_labels, predictions)))
    # print("Classification report:")
    # print(classification_report(test_labels, predictions))
    # print("Confusion matrix:")
    # print(confusion_matrix(test_labels, predictions), '\n')
