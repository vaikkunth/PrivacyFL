import time
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from warnings import simplefilter
from pyspark.ml.classification import LogisticRegression
from sklearn.datasets import load_digits
from pyspark.sql import Row
from pyspark.ml.linalg import SparseVector

simplefilter(action='ignore', category=FutureWarning)

spark = SparkSession.builder.appName('abc').getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

vec1 = SparseVector(4, [1,2,3,4])
vec2 = SparseVector(4, [1,2,3,4])

row1 = Row(label = 7, features = vec1)
row2 = Row(label = 3, features = vec2)

from pyspark.sql.types import *
from pyspark.sql import Row

schema = StructType([StructField('label', IntegerType()), StructField('features',ArrayType())])
rows = [Row(name='Severin', age=33), Row(name='John', age=48)]
df = spark.createDataFrame(rows, schema)

