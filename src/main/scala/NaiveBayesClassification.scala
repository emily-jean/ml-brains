
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Encoders
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vector

// args(0) = input to file
object NaiveBayesClassification {
  
   def main(args: Array[String]): Unit = {
     
    // Set configuration   
    val conf = new SparkConf().
      setAppName("scans")

    val sc = new SparkContext(conf)
    
    //if(args.length < 2) {
      //println("Please provide args for input and output paths")
    //}
    
    //file_contents.foreach(println) 
    val spark=SparkSession.builder().getOrCreate()
    import spark.implicits._
    
    val df = sc.textFile(args(0))
    .map(line => line.split(",").toList.map(_.toDouble))
    .map(lst => ( Vectors.dense(lst.slice(0, lst.length-2).toArray), lst(lst.length-1)))
    .toDF("features", "label")
    

    df.printSchema()
    //println(df.show())
 
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 1234L)

    
    
    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .fit(trainingData)
      
      
   val predictionAndLabels = model.transform(testData)
    .select("features", "label", "probability", "prediction")
    .map{ case Row(features: Vector, label: Double, probability: Vector, prediction: Double) =>
        (prediction, label)
      }

    println(predictionAndLabels.getClass())
    
    val testErr = predictionAndLabels.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println(s"Test Error = $testErr")
    //println(s"Learned classification forest model:\n ${cvModel.toDebugString}")
    
    //val stats = Stats(confusionMatrix(predictionsAndLabels))
    //println(stats.toString)
    
    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels.rdd)
    println(metrics.confusionMatrix)
    
    
    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")
    
    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }
    
    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }
    
    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }
    
    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }
    
    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

   }
  
}
