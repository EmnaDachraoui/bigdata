from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

# Load data (assuming amazonreviews.csv is your file)
data = spark.read.csv("amazonreviews.csv", header=True, inferSchema=True)

# Show the data structure
data.show(truncate=False)

# Data preparation: Clean and select relevant columns
data = data.select("Review Text", "Rating")

# Handle missing or special characters in Review Text (e.g., replace \N with an empty string)
data = data.withColumn("Review Text", regexp_replace("Review Text", r"\\N", ""))

# Convert Ratings to Sentiment Labels (1 for positive, 0 for negative)
data = data.withColumn("Sentiment", 
                        when(col("Rating") >= 4, 1)
                        .when(col("Rating") <= 2, 0)
                        .otherwise(None))

# Drop rows with None sentiment (i.e., Rating == 3)
data = data.na.drop(subset=["Sentiment"])

# Tokenization
tokenizer = Tokenizer(inputCol="Review Text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashing_tf = HashingTF(inputCol="filtered", outputCol="features")

# StringIndexer for labels
label_indexer = StringIndexer(inputCol="Sentiment", outputCol="label")

# Logistic Regression Model
lr = LogisticRegression(maxIter=10, regParam=0.01)

# Create a pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, label_indexer, lr])

# Fit the model
model = pipeline.fit(data)

# Make predictions
predictions = model.transform(data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Show some predictions
predictions.select("Review Text", "Sentiment", "prediction").show(truncate=False)

# Stop the Spark session
spark.stop()
