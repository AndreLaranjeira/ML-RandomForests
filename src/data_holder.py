import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Por Victor vvvv
class DataHolder:
  def __init__(self, criterion, max_features, data):
    self.criterion = criterion
    self.max_features = max_features
    self.data = data

  # Feito por André (baseado no projeto de teste realizado em sala de aula) vvvv
  def generate_data(self, name, classifiers, train_features, train_labels, test_features, test_labels, verbose=False, random_seed=193777, scoring='accuracy', test_size=0.15):
    acc_graph = []
    cv_graph = []
    for param, rf in classifiers:
      if verbose: print("Classifier: %s" % (self.get_legend(name+' at '+str(param))))

      # 10 fold cross-validation estimate:
      kfold = model_selection.KFold(n_splits = 10, random_state = random_seed)
      cv_results = model_selection.cross_val_score(rf, train_features,
                                                    train_labels, cv = kfold,
                                                    scoring = scoring)
      if verbose: print(">> Cross-validation score: %f (%f)" %
            (cv_results.mean(), cv_results.std()))

      # Test results:
      rf.fit(train_features, train_labels)
      predictions = rf.predict(test_features)
      acc = accuracy_score(test_labels, predictions)
      if verbose: print(">> Accuracy: %f\n" % (acc))
      # print("Classification report:")
      # print(classification_report(test_labels, predictions))
      # print("Confusion matrix:")
      # print(confusion_matrix(test_labels, predictions), '\n')
      acc_graph.append(acc)
      cv_graph.append(cv_results)
    return (acc_graph, cv_graph)
  # Feito por André (baseado no projeto de teste realizado em sala de aula) ^^^^

  def generate_all_data(self, train_features, train_labels, test_features, test_labels, verbose=False):
    data = {}
    for name, classifiers in self.data.items():
      data[name] = self.generate_data(name, classifiers, train_features, train_labels, test_features, test_labels, verbose=verbose)
    self.all_data = data
    return data

  def get_max_acc(self):
    while self.all_data == None:
      self.generate_all_data()
    self.max_acc = 0.0
    self.best_classifier = None
    for k, data in self.all_data.items():
      for i, acc in enumerate(data[0]):
        if acc > self.max_acc:
          self.max_acc = acc
          self.best_classifier = self.data[k][i][1]
    return (self.max_acc,self.best_classifier)

  def get_legend(self, end=''):
    return self.criterion + ', ' + self.max_features + ':' + end

  def plot_data(self, ax, name):
    if self.all_data == None:
      self.generate_all_data()
    x = [tup[0] for tup in self.data[name]]
    ax.plot(x, self.all_data[name][0], label=self.get_legend())
# Por Victor ^^^^
