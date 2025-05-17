from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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





# 1. Calcul des sentiments
sentiment_counts = predictions.groupBy("prediction").count().toPandas()

# Mapper les classes en texte (selon échelle Rating : 1-2 = négatif, 3 = neutre, 4-5 = positif)
sentiment_map = {0.0: "négatif", 1.0: "neutre", 2.0: "positif"}  # ajuster si besoin

# Si les classes ne sont pas 0/1/2, mapper manuellement selon le modèle
# Exemple : mapping via les valeurs de label
def map_sentiment(rating):
    if rating <= 2:
        return "négatif"
    elif rating == 3:
        return "neutre"
    else:
        return "positif"

# Remplacer si prédictions ne donnent pas 0/1/2 directement
predictions = predictions.withColumn("sentiment_label", predictions["label"])
predictions = predictions.withColumn("sentiment", predictions["label"].cast("int"))
mapped = predictions.select("sentiment").toPandas()
mapped["sentiment"] = mapped["sentiment"].apply(map_sentiment)

# Compter les sentiments
sentiment_counts = mapped["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment", "count"]

colors = ["red", "gray", "green"]

# 📊 Bar chart
plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts["sentiment"], sentiment_counts["count"], color=colors)
plt.title("Bar Chart : Répartition des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de commentaires")
plt.tight_layout()
plt.savefig("bar_chart_sentiment.png", dpi=300)
plt.show()

# 🥧 Pie chart
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts["count"], labels=sentiment_counts["sentiment"], autopct="%1.1f%%", colors=colors)
plt.title("Pie Chart : Répartition des sentiments")
plt.savefig("pie_chart_sentiment.png", dpi=300)
plt.show()

# 📈 Line chart
plt.figure(figsize=(6, 4))
plt.plot(sentiment_counts["sentiment"], sentiment_counts["count"], marker="o", color="blue", linestyle='-')
plt.title("Line Chart : Répartition des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de commentaires")
plt.grid(True)
plt.tight_layout()
plt.savefig("line_chart_sentiment.png", dpi=300)
plt.show()

# 🌊 Sinusoidal chart (juste pour visuel artistique)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.figure(figsize=(6, 4))
plt.plot(x, y, color="purple")
plt.title("Sinusoidal Plot (Décoratif)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.tight_layout()
plt.savefig("sinusoidal_chart.png", dpi=300)
plt.show()
# 13. Stop Spark
spark.stop()
