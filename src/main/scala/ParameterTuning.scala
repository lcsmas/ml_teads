import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegressionModel

object ParameterTuning extends App {
  import org.apache.spark.sql.SparkSession;
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.OneHotEncoder

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val model = PipelineModel.load("tmp/logistic-regression-model")
  val training = spark.read.load("tmp/training_set")
  val test = spark.read.load("tmp/test_set")

  val predictions = model.transform(test)

  predictions.printSchema()

  //println(s"fMeasureByThreshold: ${trainingSummary.fMeasureByThreshold}")
}
