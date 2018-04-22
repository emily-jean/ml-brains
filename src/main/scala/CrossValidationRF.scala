
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Encoders
//import org.apache.spark.ml.classification.LogisticRegression
 import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.ml.linalg._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row

object CrossValidationRF {
  
  
  def main(args: Array[String]): Unit = {
     
    // Set configuration   
    val conf = new SparkConf().
      setAppName("scans")

    val sc = new SparkContext(conf)
    
    //if(args.length < 2) {
      //println("Please provide args for input and output paths")
    //}
    
    
    val spark=SparkSession.builder().getOrCreate()
    import spark.implicits._
    
    // Load input file
    // Need to decide what to cache/persist
    // Change 1: File input format zip or csv/txt?
    val data = sc.textFile(args(0))
    .map(line => line.split(",").toList.map(_.toDouble))
    .map(lst => ( Vectors.dense(lst.slice(0, lst.length-2).toArray), lst(lst.length-1)))
    .toDF("features", "label")
    
    data.printSchema()
    println(data.show())
    
    // Split the data into training and test sets (30% held out for testing).
    // Change 2: Need to change this from random split
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    
    // Train a RandomForest model.
    val rf = new RandomForestClassifier()

    // Perform hyper parameterization; add/remove parameters      
    val paramGrid = new ParamGridBuilder()
    .addGrid(rf.maxDepth, Array(3, 5, 8, 10))
    .addGrid(rf.maxBins, Array(10, 20, 30))
    .addGrid(rf.numTrees, Array(5, 10, 20, 30))
    .addGrid(rf.impurity, Array("gini", "entropy"))
    .build()  
      
    val cv = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)  // Use 3+ in practice
      .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

  
    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(trainingData)
    

    val predictionAndLabels = cvModel.transform(testData)
    .select("features", "label", "probability", "prediction")
    .map{ 
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
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
