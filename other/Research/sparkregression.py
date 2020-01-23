import pyspark
sc = pyspark.SparkContext(appName='DistributedMNIST')
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
Y = digits.target

from pyspark.ml.classification import LogisticRegression
import pandas as pd


lr = LogisticRegression()
dataset =
model = lr.fit(dataset=(X, Y))