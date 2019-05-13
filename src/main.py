# Program to classify leaf data using random forests.

# Package imports:
import numpy as np
import pandas
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# User imports:
from data_holder import *

# Main function:

# Por André vvvv
start_time = time.time()

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
# Por André ^^^^

# Initializing classifiers:

# Por Victor vvvv
datas = []
for criterion in ['entropy', 'gini']:
  for max_features in ['sqrt', 'log2', None]:
    changes = {}

    # Analisar como variar a quantidade de árvores afeta
    changes['n_estimators'] = []
    for i in range(1,502,100):
      changes['n_estimators'].append((i,
                                      RandomForestClassifier(n_estimators = i,
                                                            criterion = criterion,
                                                            max_depth = None,
                                                            min_samples_split = 2,
                                                            min_samples_leaf = 1,
                                                            max_features = max_features,
                                                            max_leaf_nodes = None)))

    # Analisar poda de árvore limitando profundidade máxima (espero pouco disso)
    changes['max_depth'] = []
    for i in range(0,14,2):
      changes['max_depth'].append((i+1,
                                   RandomForestClassifier(n_estimators = 100,
                                                            criterion = criterion,
                                                            max_depth = i+1,
                                                            min_samples_split = 2,
                                                            min_samples_leaf = 1,
                                                            max_features = max_features,
                                                            max_leaf_nodes = None)))

    # Analisar poda de árvore limitando mínimo de dados para ocorrer divisão
    changes['min_samples_split'] = []
    for i in range(0,14,1):
      changes['min_samples_split'].append((i+2,
                                            RandomForestClassifier(n_estimators = 100,
                                                            criterion = criterion,
                                                            max_depth = None,
                                                            min_samples_split = i+2,
                                                            min_samples_leaf = 1,
                                                            max_features = max_features,
                                                            max_leaf_nodes = None)))

    # Analisar poda de árvore limitando mínimo de dados por folha (slides)
    changes['min_samples_leaf'] = []
    for i in range(20):
      changes['min_samples_leaf'].append((i+1,
                                          RandomForestClassifier(n_estimators = 100,
                                                            criterion = criterion,
                                                            max_depth = None,
                                                            min_samples_split = 2,
                                                            min_samples_leaf = i+1,
                                                            max_features = max_features,
                                                            max_leaf_nodes = None)))

    # Analisar poda limitando número de folhas
    changes['max_leaf_nodes'] = []
    for i in range(14, 141, 14):
      changes['max_leaf_nodes'].append((i,
                                        RandomForestClassifier(n_estimators = 100,
                                                            criterion = criterion,
                                                            max_depth = None,
                                                            min_samples_split = 2,
                                                            min_samples_leaf = 1,
                                                            max_features = max_features,
                                                            max_leaf_nodes = i)))
    datas.append(
      DataHolder(criterion, 'all' if max_features==None else max_features, changes)
      )

# Cross-validation predictions and test results:
names = [
  'n_estimators',
  'max_depth',
  'min_samples_split',
  'min_samples_leaf',
  'max_leaf_nodes'
  ]

for data in datas:
  data.generate_all_data(train_features, train_labels, test_features, test_labels, verbose=True)
  max_acc, best_classifier = data.get_max_acc()
for data in datas:
  print('Best Accuracy for ' + data.get_legend(': '+str(max_acc)) )
  print('Parameters:')
  for name in names:
    print(name + ' = ' + str(getattr(best_classifier,name)))
  print()

for name in names:
  fig = plt.figure(name)
  ax = fig.gca()
  for data in datas:
    data.plot_data(ax, name)
  ax.legend(loc='best')
  ax.set_xlabel('Value of the parameter ' + name)
  ax.set_ylabel('Accuracy')
  fig.savefig('img/'+name+'.png')

end_time = time.time()
print('Execution time = {0:.0f} seconds'.format(end_time-start_time))

plt.show()
# Por Victor ^^^^
