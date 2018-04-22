import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.tuning.TrainValidationSplitModel

object FinalClassification {

  def main(args: Array[String]): Unit = {

    val raw_input_path 		= args(0)
    val preproc_out_path	= args(1)
    val training_path 		= args(2)		// Used Here X
    val testing_path 		= args(3)		// Used Here X
    val validation_path 	= args(4)
    val output_path 		= args(5)
    val model_in_path 		= args(6)
    val model_out_path 		= args(7)		// Used Here X
    val metrics_file_path = args(8)		// Used Here X
    val max_depth 			= args(9)		// Used Here X
    val max_bins 			= args(10)	// Used Here X
    val num_trees 			= args(11)	// Used Here X
    val partition_size    = args(12)	// Used Here X 


    // Set configuration   

    val conf = new SparkConf().
    setAppName("classification")

    val sc = new SparkContext(conf)
    
    val spark=SparkSession.builder().getOrCreate()
    import spark.implicits._
    
    val validationData = sc.textFile(validation_path, partition_size.toInt).
      map(line => line.split(",").toList.map(_.toDouble)).
      map(lst => ( Vectors.dense(lst.slice(0, lst.length-2).toArray), lst(lst.length-1))).toDF("features", "label")

    val model = TrainValidationSplitModel.load(model_in_path).bestModel
    println("Model loaded successfully")

    // Make predictions.
    val predictions = model.transform(validationData)

    // Save
    predictions.select("prediction").map{r => 
      r.getDouble(0).toInt
    }.rdd.saveAsTextFile(output_path)
//    val pred = predictions.select("label").rdd
//    pred.repartition(1).saveAsTextFile(output_path)
  }
}
