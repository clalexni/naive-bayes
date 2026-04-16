# naive bayes classification

~~~
./naive_bayes.py data/train.dat data/test.dat > out.txt
~~~

## Personal Notes: 
- what is naive bayes?
  - use conditional independence assumption to do classification
- what to keep track of?
  - count instances of each class value in col_values[-1]
  - count instances of attribute value given a class value in col_values[i] from i = 0 to -2
- what to do after that?
  - translate count to probability or cond prob
  - implement argmax




