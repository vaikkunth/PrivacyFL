import pyspark

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('abc').getOrCreate()
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
import sklearn
import pandas as pd
import numpy as np
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, IntegerType

from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

#todo: note difficulties of spark: cannot set weights for evaluation so using sklearn slows that down

# todo: make the lists wayyyy bigger to show the computation time is faster on spark
# todo: play with partition size/parameters for each dataset
# todo: sanity checks (same weights produced with no DP noise if non-random train test slit)
# todo: update python code to include intercept


X = X.tolist()
y = y.tolist()

data = zip(y,X)

formatted = [(int(y_i), DenseVector(x_i)) for y_i, x_i in data]
fields = [StructField('label', IntegerType(), True), StructField('features', VectorUDT(), True)]
schema = StructType(fields)
data = spark.createDataFrame(formatted, schema)

#train_data, test_data = data.randomSplit([1/2, 1/2])
train_data = data


lr = LogisticRegression(maxIter=10)
lrModel = lr.fit(train_data)
trainingSummary = lrModel.summary
print(trainingSummary.accuracy)
######

result = lrModel.transform(train_data) # test_data
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
print("Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))
coefficientMatrix = lrModel.coefficientMatrix
intercepts = lrModel.interceptVector
print(intercepts)

import sklearn.linear_model
lrSK = sklearn.linear_model.LogisticRegression()
X, y = digits.data, digits.target
lrSK.fit(X[:10], y[:10])
lrSK.coef_ = coefficientMatrix.toArray()
lrSK.intercept_ = intercepts.toArray()
print(lrSK.score(X, y))

#sc = pyspark.SparkContext.getOrCreate()
#session = SparkSession(sc)