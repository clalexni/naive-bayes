# naive bayes classification

## Goal
- Learn from data using naive bayes classifier
- output learn result and accuracy using test set

## Requirements
- install Python 3.6
- data file (path) for inputs training and test set
- command line: 
  - run these two lines to ensure the correct python version
  - the second line takes two input of data file path (./data/train.dat for instance)
    - the first input is the training set and the second input should be the test set
~~~
chmod u+x naive_bayes.py
~~~
~~~
./naive_bayes.py [input1] [input2]
~~~

- for instance: 
~~~
./naive_bayes.py data/train.dat data/test.dat
~~~
- use the following command to redirect output, for example:
~~~
./naive_bayes.py data/train.dat data/test.dat > out.txt
~~~
- tested on macos/Linux


## Personal Notes: 
- what is naive bayes?
  - use conditional independence assumption to do classification
- what to keep track of?
  - count instances of each class value in col_values[-1]
  - count instances of attribute value given a class value in col_values[i] from i = 0 to -2
- what to do after that?
  - translate count to probability or cond prob
  - implement argmax




