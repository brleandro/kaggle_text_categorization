This script predicts topics of comments on Reddit. A training set must be informed so that it can use this data set to
predict the topics of the comments of a given test data set.

To execute the scrit bag_of_words.py you should informe the following parameters:

-i : the file path for the training data
-t : the file path for the test data which will have the topics predicted
-a : hyperparemeter alpha Laplace Smoothing
-o : output file name where the prediction will be registered

Example:

python bag_of_words.py -i /opt/data_train.pkl -t /data_test.pkl -a 0.4 -o ./submission2.csv