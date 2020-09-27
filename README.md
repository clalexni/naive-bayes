# naive bayes classification

## Goal
Learn from data using naive bayes classifier

## Requirements
- install Python 3.6
- data file (path) for inputs training and test set
- command line: 
  - run these two lines to ensure the correct python version
  - the first line gives user permission to execute the code using shebang style
  - the second line takes two input of data file path (./data/train.dat for instance)
  - the first input is the training set and the second input should be the test set
~~~
chmod u+x main.py
~~~
~~~
./main.py [input1] [input2]
~~~

- for instance: 
~~~
./main.py data/train.dat data/test.dat
~~~
- use the following command to redirect output, for example:
~~~
./main.py data/train.dat data/test.dat > out.txt
~~~
- tested on macos/Linux


## Personal Note: 
- count instances of each class value in col_values[-1]
- count instances of attribute value given a class value in col_values[i] from i = 0 to -2
- translate count to probability or cond prob
- implement argmax




