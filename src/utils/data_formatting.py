from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.types import StructType, StructField, IntegerType

def create_spark_df(X, y):
    """
    Used to convert train data and test data into Spark dataframes for training
    :param X: numpy array of data
    :param y: numpy array of labels
    :return: pyspark dataframe with 'features' and 'label' columns
    """
    spark = SparkSession.builder.getOrCreate()

    X = X.tolist()
    y = y.tolist()
    data = zip(y, X)

    formatted = [(int(y_i), DenseVector(x_i)) for y_i, x_i in data]
    fields = [StructField('label', IntegerType(), True), StructField('features', VectorUDT(), True)]
    schema = StructType(fields)
    data = spark.createDataFrame(formatted, schema)
    return data
