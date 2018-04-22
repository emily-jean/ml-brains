
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Encoders
//import org.apache.spark.ml.classification.LogisticRegression
 import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.functions.udf
//import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.sql.Row

import org.apache.spark.SparkFiles
import scala.io.Source
import java.io._

object ModelTraining {
  
  def main(args: Array[String]): Unit = {

		val raw_input_path 		= args(0)
		val preproc_out_path	= args(1)
		val training_path 		= args(2)		  // Used Here X
		val testing_path 			= args(3)		// Used Here X
		val validation_path 	= args(4)
		val output_path 			= args(5)
		val model_in_path 		= args(6)
		val model_out_path 		= args(7)		  // Used Here X
		val metrics_file_path = args(8)		  // Used Here X
		val max_depth 				= args(9)		  // Used Here X
		val max_bins 					= args(10)  	// Used Here X
		val num_trees 				= args(11)	    // Used Here X
        val partition_size    = args(12)	// Used Here X
     
    // Set configuration   
    val conf = new SparkConf().
      setAppName("scans")

    val sc = new SparkContext(conf)
    
    /*
    if(args.length < 2) {
      println("Please provide args for input and output paths")
    }
    */
    
    val spark=SparkSession.builder().getOrCreate()
    import spark.implicits._
    
    // Load input file
    // Need to decide what to cache/persist
    val trainingData = sc.textFile(training_path, minPartitions=partition_size.toInt)
    .map(line => line.split(",").toList.map(_.toDouble))
    .map(lst => ( Vectors.dense(lst.slice(0, lst.length-2).toArray), lst(lst.length-1)))
    .toDF("features", "label").persist()
    
    //trainingData.printSchema()
    //println(trainingData.show())
    
    //--------------------------------------------------------------------------------------------------------------------
    // TRAINING
    //--------------------------------------------------------------------------------------------------------------------
    
    val rf = new RandomForestClassifier()
    
    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.Perform hyper parameterization; add/remove parameters
    val paramGrid = new ParamGridBuilder()
    .addGrid(rf.maxDepth, Array(max_depth.toInt))
    .addGrid(rf.maxBins, Array(max_bins.toInt))
    .addGrid(rf.numTrees, Array(num_trees.toInt))
    .addGrid(rf.impurity, Array("gini"))
    .build()  
    
   
    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(rf)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 100% of the data will be used for training and the remaining 0% for validation.
      .setTrainRatio(1.0)
    
    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(trainingData)
    
    
    //--------------------------------------------------------------------------------------------------------------------
    // TESTING
    //--------------------------------------------------------------------------------------------------------------------
    
    val testData = sc.textFile(testing_path)
    .map(line => line.split(",").toList.map(_.toDouble))
    .map(lst => ( Vectors.dense(lst.slice(0, lst.length-2).toArray), lst(lst.length-1)))
    .toDF("features", "label")//.persist()

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    val predictionAndLabels = model.transform(testData)
    .select("features", "label", "probability", "prediction")
    .map{ 
      case Row(features: Vector, label: Double, probability: Vector, prediction: Double) =>
        (prediction, label)
      }

    //println(predictionAndLabels.getClass())
    
    val metricOutput = new StringBuilder()

    metricOutput.append("Model Parameters: " + "\n")
    metricOutput.append("Max Depth: " + max_depth.toInt + "\n")
    metricOutput.append("Max Bins: " + max_bins.toInt +"\n")
    metricOutput.append("Number of Trees: " + num_trees.toInt + "\n")
    metricOutput.append("\n")


    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels.rdd)
    //println(metrics.confusionMatrix)
    metricOutput.append("\nConfusion Matrix >>>>\n")
    metricOutput.append(metrics.confusionMatrix + "\n")
    
    
    // Overall Statistics
    val accuracy = metrics.accuracy
    //println("Summary Statistics")
    //println(s"Accuracy = $accuracy")
    metricOutput.append("\nSummary Statistics >>>>>\n")
    metricOutput.append(s"Accuracy = $accuracy" + "\n")

    val testErr = predictionAndLabels.filter(r => r._1 != r._2).count.toDouble / testData.count()
    //println(s"Test Error = $testErr")
    metricOutput.append(s"Test Error = $testErr" + "\n")
    //println(s"Learned classification forest model:\n ${model.toDebugString}")
    
    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      //println(s"Precision($l) = " + metrics.precision(l))
      metricOutput.append("Precision($l) = " + metrics.precision(l) + "\n")
    }
    
    // Recall by label
    labels.foreach { l =>
      //println(s"Recall($l) = " + metrics.recall(l))
      metricOutput.append("Recall($l) = " + metrics.recall(l) + "\n")
    }
    
    // False positive rate by label
    labels.foreach { l =>
      //println(s"FPR($l) = " + metrics.falsePositiveRate(l))
      metricOutput.append("FPR($l) = " + metrics.falsePositiveRate(l) + "\n")
    }
    
    // F-measure by label
    labels.foreach { l =>
      //println(s"F1-Score($l) = " + metrics.fMeasure(l))
      metricOutput.append("F1-Score($l) = " + metrics.fMeasure(l) + "\n")
    }
    
    // Weighted stats
    //println(s"Weighted precision: ${metrics.weightedPrecision}")
    metricOutput.append(s"Weighted precision: ${metrics.weightedPrecision}" + "\n")
    //println(s"Weighted recall: ${metrics.weightedRecall}")
    metricOutput.append(s"Weighted recall: ${metrics.weightedRecall}" + "\n")
    //println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    metricOutput.append(s"Weighted F1 score: ${metrics.weightedFMeasure}" + "\n")
    //println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
    metricOutput.append(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}" + "\n")
    
    //sc.parallelize(metricOutput, 1).saveAsTextFile(metrics_file_path)
    //metricOutput.saveAsTextFile(metrics_file_path)

    //val dir: File = metrics_file_path.toFile.createIfNotExists(true) 

    // Save and load model
    model.save(model_out_path)
    
    sc.parallelize(List(metricOutput.toString()),1).saveAsTextFile(metrics_file_path)
    
  }
  
}
