from pyspark.sql import SparkSession
spark = SparkSession.builder\
       .appName("Classification")\
       .getOrCreate()
sc = spark.sparkContext
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# spark.sparkContext._conf.getAll()

from pyspark import SparkConf
from pyspark.sql.functions import lit, array_remove

from pyspark.sql.functions import rand

from pyspark.ml.feature import OneHotEncoder
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array

from pyspark.sql.functions import col, countDistinct

from pyspark.sql import SparkSession

heart_df = spark.read.csv('/content/heart.csv', sep=',', inferSchema=True, header=True)

# o2s_df = spark.read.csv('/content/o2Saturation.csv', sep=',',
#                          inferSchema=True, header=True)

# heart_df.show(5)

# heart_df.summary()

# from pyspark.sql.functions import col,isnan,when,count
# df2 = heart_df.select([count(when(col(c).contains('None') | \
#                             col(c).contains('NULL') | \
#                             (col(c) == '' ) | \
#                             col(c).isNull() | \
#                             isnan(c), c 
#                            )).alias(c)
#                     for c in heart_df.columns])
# df2.show()

# expression = [countDistinct(c).alias(c) for c in heart_df.columns]
# heart_df.select(*expression).show()

# heart_df.cube('output').count().show()
# heart_df.cube('cp').count().show()
# heart_df.cube('fbs').count().show()
# heart_df.cube('restecg').count().show()
# heart_df.cube('exng').count().show()
# heart_df.cube('slp').count().show()
# heart_df.cube('caa').count().show()
# heart_df.cube('thall').count().show()

def one_hot_encoder(dataframe, column_name, output_column_name, alias_name):
  single_col_ohe = OneHotEncoder(inputCol=column_name, outputCol=output_column_name)
  single_col_model = single_col_ohe.fit(heart_df).transform(heart_df)
  df_col_onehot = single_col_model.select('*', vector_to_array(output_column_name).alias(alias_name))
  
  num_categories = len(df_col_onehot.first()[alias_name])

  cols_expanded = [(F.col(alias_name)[i]) for i in range(num_categories)]
  df_cols_onehot = df_col_onehot.select(*cols_expanded)
  df_cols_onehot = df_cols_onehot.select( df_cols_onehot.columns[:-1])
  dataframe = dataframe.drop(column_name)

  dataframe.createOrReplaceTempView("heart_view")
  df_cols_onehot.createOrReplaceTempView("df3_view")
  num_of_catches = spark.sql("SELECT * from heart_view inner join df3_view")
  return num_of_catches

df3 = one_hot_encoder(heart_df, "slp", "new_slp", "new_slp_value")

df3 = one_hot_encoder(df3, "cp", "new_cp", "new_cp_value")
df3 = one_hot_encoder(df3, "restecg", "new_restecg", "new_restecg_value")
df3 = one_hot_encoder(df3, "caa", "new_caa", "new_caa_value")
df3 = one_hot_encoder(df3, "thall", "new_thall", "new_thall_value")

# df3.show(5)

# splits = df3.randomSplit([0.8, 0.2])

# splits[0].count()

input_columns = ["age", "sex", "trtbps", "chol", "fbs", "thalachh", "exng",\
"oldpeak", "new_slp_value[0]", "new_cp_value[0]", "new_cp_value[1]", "new_restecg_value[0]",\
"new_caa_value[0]", "new_caa_value[1]", "new_caa_value[2]", "new_thall_value[0]", "new_thall_value[1]"]

# input_columns = ["age", "sex", "trtbps", "chol", "fbs", "thalachh", "exng",\
# "oldpeak", "new_slp_value[0]", "new_cp_value[0]", "new_cp_value[1]", "new_restecg_value[0]",\
# "new_caa_value[0]", "new_caa_value[1]", "new_caa_value[2]", "new_thall_value[0]", "new_thall_value[1]"]

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=input_columns, outputCol="features")
df4 = assembler.transform(df3)

df4 = df4.select('features', 'output')

df4.show()