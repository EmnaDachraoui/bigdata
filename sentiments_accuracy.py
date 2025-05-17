from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialiser la session Spark
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Charger les données CSV avec header et détection automatique du schéma
data = spark.read.csv("amazonreviews.csv", header=True, inferSchema=True)

# 3. Afficher les colonnes pour vérifier
print("Colonnes disponibles :", data.columns)

# 4. Sélectionner les colonnes texte et label (ici 'Review Text' et 'Rating')
df = data.select(
    col("Review Text").alias("text"),
    col("Rating").cast("double").alias("label")  # Conversion en double
).dropna()

# 5. Afficher un aperçu des données
df.show(5, truncate=80)

# 6. Diviser les données en train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 7. Construire le pipeline ML
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# 8. Entraîner le modèle sur les données d'entraînement
model = pipeline.fit(train_df)

# 9. Faire des prédictions sur le jeu de test
predictions = model.transform(test_df)

# 10. Afficher les résultats (texte, label réel, prédiction, probabilités)
predictions.select("text", "label", "prediction", "probability").show(10, truncate=80)

# 11. Évaluer la précision
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Précision du modèle : {accuracy:.4f}")

# 12. Arrêter la session Spark
spark.stop()
