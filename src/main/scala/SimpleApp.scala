/**
  * Created by GosiaGosia on 2017-04-06.
  */

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


object SimpleApp {
  def main(args: Array[String]) {
   // val logFile = "C:\\Users\\GosiaGosia\\Desktop\\nowy.txt" // Should be some file on your system
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder
      .appName("SimpleApp")
      .getOrCreate()


    //credits.printSchema()

   val credits = spark.read.option("header","true").option("inferSchema","true").csv("german_credit.csv")


   val nonFeatureCols = Array("Creditability")

    val featureCols=credits.columns.diff(nonFeatureCols);

    //creditsDbl.show(10)

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val finalCreditData = assembler.transform(credits)

    val Array(training,test)=finalCreditData.randomSplit(Array(0.7,0.3))

    println(training.count())
    println(test.count())

    val rf = new RandomForestClassifier()
      .setLabelCol("Creditability")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val rfmodel = rf.fit(training)

    val predictions = rfmodel.transform(test)

    predictions.select("Creditability", "prediction", "features").show(100)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Creditability")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " +  accuracy)


    spark.stop()

    sc.stop()
  }

}
