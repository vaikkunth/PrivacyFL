import pyspark
from pyspark import SparkContext
import numpy as np
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

import pandas as pd
sc = SparkContext("local", "First App")
df = pd.read_csv('adult_clean.csv', usecols = (0, 2, 4,10,11, 12, 14))
# todo prepare df
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
pipeline = Pipeline(stages=[lr])
lrModel = pipeline.fit(df)
