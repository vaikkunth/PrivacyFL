import pandas as pd


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('abc').getOrCreate()

df_training = (spark
               .read
               .options(header = False, inferSchema = True)
               .csv("mnist_test.csv"))
columns = ["_c" + str(i+1) for i in range(784)]
from pyspark.ml.feature import VectorAssembler

vectorizer = VectorAssembler(inputCols=columns, outputCol="features")
training = (vectorizer
            .transform(df_training)
            .select("_c0", "features")
            .toDF("label", "features")
            .cache())
print(training.Sschema)