from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# 1. Initialiser la session Spark
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

# 2. Charger les données depuis le fichier CSV
try:
    df_csv = spark.read.csv("amazonreviews.csv", header=True, inferSchema=True) \
        .select("Review Text", "Rating") \
        .withColumn("Review Text", regexp_replace("Review Text", r"\\N", ""))
    print("📂 Fichier 'amazonreviews.csv' chargé avec succès.")
except Exception as e:
    print("⚠️ Erreur lors du chargement du fichier CSV :", e)
    df_csv = None

# 3. Ajouter des données manuelles avec Pandas
sample_data = pd.DataFrame({
    "Review Text": [
        "This is an excellent product I bought this for my digital camera It has a great battery life per charge Charger is handy to carry",
        "Terrible product. Stopped working after one use.",
        "Average performance, nothing special.",
        "Fantastic sound quality and easy to use.",
        "Battery life is terrible and charger stopped working."
    ],
    "Rating": [5, 1, 3, 5, 1]
})
df_manual = spark.createDataFrame(sample_data)

# 4. Fusionner CSV + données manuelles
data = df_csv.union(df_manual) if df_csv else df_manual

# 5. Nettoyage et transformation : étiquettes de sentiment
data = data.withColumn("Sentiment",
    when(col("Rating") >= 4, 1)
    .when(col("Rating") <= 2, 0)
    .otherwise(None)
)

# 6. Supprimer les lignes neutres (rating == 3 ou nuls)
data = data.na.drop(subset=["Sentiment", "Review Text"])

# 7. Définir les étapes du pipeline NLP + ML
tokenizer = Tokenizer(inputCol="Review Text", outputCol="words")
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashing_tf = HashingTF(inputCol="filtered", outputCol="features")
label_indexer = StringIndexer(inputCol="Sentiment", outputCol="label")
lr = LogisticRegression(maxIter=10, regParam=0.01)

pipeline = Pipeline(stages=[
    tokenizer,
    stopwords_remover,
    hashing_tf,
    label_indexer,
    lr
])

# 8. Entraîner le modèle
model = pipeline.fit(data)

# 9. Prédire sur les données
predictions = model.transform(data)

# 10. Évaluer la précision
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"\n📊 Accuracy du modèle : {accuracy:.2f}")

# 11. Afficher les prédictions
print("\n🔍 Quelques prédictions :")
predictions.select("Review Text", "Sentiment", "prediction").show(truncate=False)

# 12. Exporter les résultats vers un fichier CSV
predictions.select("Review Text", "Sentiment", "prediction") \
    .toPandas() \
    .to_csv("sentiment_predictions.csv", index=False)
print("✅ Résultats exportés dans 'sentiment_predictions.csv'")

# 13. Fermer la session Spark proprement spark.stop()
