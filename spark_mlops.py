# Install spark

!pip install pyspark
!pip install findspark

# Start session

import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("E-commerce Search Terms").getOrCreate()

# Download The search term dataset from the below url
# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DB0321EN-SkillsNetwork/Bigdata%20and%20Spark/searchterms.csv

import subprocess

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DB0321EN-SkillsNetwork/Bigdata%20and%20Spark/searchterms.csv"
subprocess.run(["wget", url, "-O", "searchterms.csv"])

# Load the csv into a spark dataframe

df = spark.read.csv("searchterms.csv", header=True, inferSchema=True)

# Print the number of rows and columns

print((df.count(), len(df.columns)))

# Print the top 5 rows

df.show(5)

# Find out the datatype of the column searchterm

print(df.schema['searchterm'].dataType)

# How many times was the term `gaming laptop` searched?

gaming_laptop_count = df.filter(df.searchterm == 'gaming laptop').count()
print(f"'gaming laptop' was searched {gaming_laptop_count} times.")

# Print the top 5 most frequently used search terms

top_terms = df.groupBy('searchterm').count().orderBy('count', ascending=False).limit(5)
top_terms.show()

# The pretrained sales forecasting model is available at  the below url
# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DB0321EN-SkillsNetwork/Bigdata%20and%20Spark/model.tar.gz

# Load the sales forecast model.

import tarfile
import urllib.request

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DB0321EN-SkillsNetwork/Bigdata%20and%20Spark/model.tar.gz"
tar_file_path = "model.tar.gz"

urllib.request.urlretrieve(url, tar_file_path)

with tarfile.open(tar_file_path, "r:gz") as tar:
    tar.extractall()
    
# Using the sales forecast model, predict the sales for the year of 2023.

from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("SalesForecast").getOrCreate()
model_path = "sales_prediction.model"
model = LinearRegressionModel.load(model_path)

data = spark.createDataFrame([(2023,)], ["Year"])
assembler = VectorAssembler(inputCols=["Year"], outputCol="features")
data = assembler.transform(data)

predictions = model.transform(data)
predictions.select("Year", "features", "prediction").show()