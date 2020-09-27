#!/usr/bin/env python3.6
import sys
from itertools import islice
import math


class DataSet:
  """
    dataset fields:
    d.examples      An example matrix. It is basically a list of examples, 
                    each example is a list of attribute values + class value
    d.col_indices   A list of indices that is corresponded to a column of the 
                    examples matrix
    d.col_names     A list of name (label) corresponding to the matrix's column
    d.attr_indices  Same as col_indices but without the last column (class).
                    In another word, each attribute is a column index
    d.col_values    A list of list: each sublist is the set of possible values
                    corresponding to an attribute or class
  """
  def __init__(self, examples=None, col_names=None):
    self.examples = examples
    self.col_names = col_names
    self.col_indices = list(range(len(self.examples[0])))
    self.attr_indices = [i for i in self.col_indices[:-1]]
    self.col_values = list(map(lambda x: set(x), zip(*self.examples)))

def naive_bayes_learner(dataset):
  """
  return probability of classes and conditional probablility of
  attribute value given a class value in dictionary format
  """
  def naive_bayes_learning(examples, col_values):

    # dictionary of count of attr value given class value
    cond_prob = {c: [ {a: 0 for a in A} for A in col_values[0:-1]]
                 for c in col_values[-1]} 

    # dictionary of count of class where key is class
    prob = {k: 0 for k in col_values[-1]} 

    for ex in examples:
      prob[ex[-1]] += 1
      for i in range(len(ex)-1):
        cond_prob[ex[-1]][i][ex[i]] += 1

    # change from count to prob
    for (c, v) in cond_prob.items():
      for A in v:
        for k in A.keys():
          count = A[k]
          A[k] = count/prob[c]
    for c in prob:
      count = prob[c]
      prob[c] = count/len(examples)
          
    #print('cond_prob')
    #for (k, v) in cond_prob.items():
    #  print(k, ': ', v)
    #print('prob')
    #for (k, v) in prob.items():
    #  print(k,': ', v)
    return prob, cond_prob
  return naive_bayes_learning(dataset.examples, dataset.col_values)

def parse_data(input_file):
  """ 
  return a list of column names and an examples matrix
  """
  with open(input_file, 'r') as file:
    data = file.readlines()
    col_names = data[0].split()
    examples = [line.split() for line in islice(data, 1, len(data))]
    return col_names, examples

def stdout(prob, cond_prob, labels):
  """
  print learn result
  """
  for (c, v) in cond_prob.items():
    print('P(' + labels[-1] + '=' + c + ')=' + '{:.2f}'.format(prob[c]), end=' ')
    index = 0
    for A in v:
      for k in A.keys():
        print('P(' + labels[index] + '=' + k + '|' + labels[-1] + ')=' + 
              '{:.2f}'.format(A[k]), end=' ')
      index += 1
    print()

def argmax(prob, cond_prob, example):
  """
  return argmax aka class value
  """
  c, lv = None, float('-inf') # c is class
  for (k, v) in prob.items():
    value, index = 0, 0
    for attr_value in example[0:-1]:
      value += math.log10(cond_prob[k][index][attr_value])
      index += 1
    value += math.log10(prob[k])
    if value > lv:
      lv = value
      c = k
  return c


def test_accuracy(prob, cond_prob, test_examples):
  """
  test the learning result using testing set and return accuracy
  """

  count = 0
  for ex in test_examples:
    if ex[-1] == argmax(prob, cond_prob, ex):
      count += 1
  return count/len(test_examples) # return accuracy


if __name__ == '__main__':
  train = sys.argv[1]
  test = sys.argv[2]

  col_names, examples = parse_data(train)
  train_ds = DataSet(examples, col_names)

  prob, cond_prob = naive_bayes_learner(train_ds)
  stdout(prob, cond_prob, train_ds.col_names)

  print('\nAccuracy on training set (' + str(len(examples)) +
        '): ' + '{:.2%}'.format(test_accuracy(prob, cond_prob, examples)))

  test_examples = parse_data(test)[1]
  print('Accuracy on test set (' + str(len(test_examples)) +
        '): ' + '{:.2%}'.format(test_accuracy(prob, cond_prob, test_examples)))

