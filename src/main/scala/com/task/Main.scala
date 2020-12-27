package com.task

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.sql.expressions.Window


object Main extends App {
  override def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("myApp")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val data = spark.read.json("src/data/DO_record_per_line.json")
    val cleaned_df = data
      .withColumn("desc", regexp_replace(col("desc"), "[^\\w\\sа-яА-ЯЁё]", ""))
      .withColumn("desc", lower(trim(regexp_replace(col("desc"), "\\s+", " "))))
      .where(length(col("desc")) > 0)

    val tokenizer = new Tokenizer().setInputCol("desc").setOutputCol("tokens")
    val tokensData = tokenizer.transform(cleaned_df)

    val hashingTF = new HashingTF()
      .setInputCol("tokens").setOutputCol("rawFeatures").setNumFeatures(10000)

    val featurizedData = hashingTF.transform(tokensData)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
      .setMinDocFreq(2)

    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    val sparseToDense = udf((v: Vector) => v.toDense)
    val newDf = rescaledData
      .withColumn("features", sparseToDense(col("features")))

    val cosSimilarity = udf { (x: Vector, y: Vector) =>
      val vec1 = x.toArray
      val vec2 = y.toArray
      val l1 = scala.math.sqrt(vec1.map(x => x*x).sum)
      val l2 = scala.math.sqrt(vec2.map(x => x*x).sum)
      val scalar = vec1.zip(vec2).map(p => p._1*p._2).sum
      scalar/(l1*l2)
    }

    val id_list = Seq(59, 4, 46, 801, 1433, 2001)

    val target_df = newDf
      .filter(col("id").isin(id_list: _*))
      .select(col("id").as("target_id"), col("features").as("target_features"),
        col("lang").as("target_lang"))

    val joinedDf = newDf.join(broadcast(target_df),
        newDf("id") =!= target_df("target_id") &&
        newDf("lang") === target_df("target_lang")
    )
      .withColumn("cosine_sim", cosSimilarity(col("features"), col("target_features")))

    val window = Window.partitionBy(col("target_id"))
      .orderBy(col("cosine_sim").desc, col("name").asc, col("id").asc)
    val filtered = joinedDf
      .withColumn("cosine_sim", when(col("cosine_sim").isNaN, 0).otherwise(col("cosine_sim")))
      .withColumn("rank", rank().over(window).alias("rank"))
      .filter(col("rank")between(1,10))

    val result = filtered.groupBy(col("target_id"))
      .agg(map(filtered.col("target_id"), collect_list(filtered.col("id"))).alias("matches"))
      .orderBy(col("target_id"))
      .repartition(1)
      .select(to_json(col("matches")))
      .write.option("header", false)
      .mode("append")
      .text("src/data/results")

    spark.close()
    spark.stop()

  }
}
