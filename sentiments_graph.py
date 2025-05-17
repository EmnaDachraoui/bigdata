from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

# 1. Initialiser la session Spark
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Charger les données CSV
data = spark.read.csv("amazonreviews.csv", header=True, inferSchema=True)
print("Colonnes disponibles :", data.columns)

# 3. Sélectionner et nettoyer les colonnes utiles
df = data.select(
    col("Review Text").alias("text"),
    col("Rating").cast("double").alias("label")
).dropna()

# 4. Afficher un aperçu
df.show(5, truncate=80)

# 5. Diviser en train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 6. Construire le pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# 7. Entraîner le modèle
model = pipeline.fit(train_df)

# 8. Prédictions
predictions = model.transform(test_df)
predictions.select("text", "label", "prediction", "probability").show(10, truncate=80)

# 9. Évaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Précision du modèle : {accuracy:.4f}")

# 10. Analyser les prédictions en sentiments
# Ajouter une colonne 'sentiment' selon la prédiction
sentiment_df = predictions.withColumn(
    "sentiment",
    when(col("prediction") <= 2, "Negatif")
    .when(col("prediction") == 3, "Neutre")
    .when(col("prediction") >= 4, "Positif")
)

# 11. Compter chaque type de sentiment
sentiment_counts = sentiment_df.groupBy("sentiment").count().toPandas()

# 12. Afficher les courbes avec matplotlib
plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts["sentiment"], sentiment_counts["count"], color=["red", "gray", "green"])
plt.title("Répartition des sentiments prédits")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de commentaires")
plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts["sentiment"], sentiment_counts["count"], color=["red", "gray", "green"])
plt.title("Répartition des sentiments prédits")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de commentaires")
plt.tight_layout()

# 🔽 Enregistrer l'image dans le dossier courant
plt.savefig("sentiment_distribution.png", dpi=300)

# 🔽 Afficher l'image à l'écran
plt.show()

# 13. Stop Spark
spark.stop()
