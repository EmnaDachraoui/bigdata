from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

# 1. Initialiser la session Spark
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Charger les données CSV avec header
data = spark.read.csv("amazonreviews.csv", header=True, inferSchema=True)

# 3. Vérifier les colonnes disponibles
print("Colonnes disponibles :", data.columns)

# 4. Sélectionner uniquement les colonnes texte et label
#    Ici 'Review Text' est le texte, 'Rating' est la note (label)
df = data.select(
    col("Review Text").alias("text"),
    col("Rating").cast("double").alias("label")  # s'assurer que label est float/double
).dropna()

df.show(5, truncate=80)

# 5. Construire le pipeline ML

tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# 6. Entraîner le modèle
model = pipeline.fit(df)

# 7. Faire des prédictions sur les mêmes données (exemple)
predictions = model.transform(df)
predictions.select("text", "label", "prediction", "probability").show(10, truncate=80)

# 8. Stopper la session Spark à la fin
spark.stop()
