import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.stat.{ChiSquareTest, Correlation}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{avg, col, count, dayofmonth, dayofweek, dayofyear, grouping, hour, max, minute, month, sum, to_timestamp, weekofyear, when, year}
import org.apache.spark.sql.types.StructType

object Test extends App{
  import org.apache.spark.sql.SparkSession;
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.OneHotEncoder
  import org.apache.spark.sql.functions.countDistinct

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val df = spark.read.load("tmp/transformed-train")

  val Array(newDf, trash) = df.randomSplit(Array(0.3,0.7))
  val aggregatedByPlacementId = newDf.groupBy($"placementId")
    .agg(avg($"isSoldInt").as("averageHasSoldByPlacementId"))

  println(trash.count())
  val joined = trash.join(aggregatedByPlacementId, Seq("placementId"),"left")

  joined.printSchema()
  joined.summary().show()

  spark.close()

}
