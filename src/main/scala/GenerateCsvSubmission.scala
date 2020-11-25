import GenerateTransformedDF.spark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.functions.{dayofmonth, dayofweek, dayofyear, month, weekofyear, year}

object GenerateCsvSubmission extends App{
  import org.apache.spark.sql.DataFrame
  import org.apache.spark.sql.functions.{hour, minute, to_timestamp}
  import org.apache.spark.sql.SparkSession;

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  def withTimeInfo(df: DataFrame): DataFrame = {
    val res = df
      .withColumn("dayOfWeek", dayofweek(to_timestamp($"timestamp".cast("Int"))).cast("String"))
      .withColumn("dayOfMonth", dayofmonth(to_timestamp($"timestamp".cast("Int"))).cast("String"))
      .withColumn("dayOfYear", dayofyear(to_timestamp($"timestamp".cast("Int"))).cast("String"))
      .withColumn("month", month(to_timestamp($"timestamp".cast("Int"))).cast("String"))
      .withColumn("year", year(to_timestamp($"timestamp".cast("Int"))).cast("String"))
      .withColumn("weekOfYear", weekofyear(to_timestamp($"timestamp".cast("Int"))).cast("String"))
      .withColumn("hour", hour(to_timestamp($"timeStamp".cast("Int"))).cast("String")).drop("timeStamp")
    if(df.columns.contains("isSold")) {
      return res
        .withColumn("isSoldInt", ($"isSold" === "true").cast("Int")).drop("isSold")
    }
    res
  }

  val test = spark.read
    .format("csv")
    .option("header", "true")
    .csv("data/test.csv")
    .toDF()
    .transform(withTimeInfo)

  val train = spark.read.load("tmp/transformed-train")
  //val aggByPlacement = train.select("placementId","averageHasSoldByPlacementId").distinct()
  //val aggByCountry = train.select("country","averageHasSoldByCountry").distinct()

  //val testWithAvgByPlacement = test.join(aggByPlacement, Seq("placementId"), "left")
  //val testWithAllAvg = testWithAvgByPlacement.join(aggByCountry, Seq("country"), "left")

  val pipeline = PipelineModel.load("tmp/fitted-pipeline")
  val model = LogisticRegressionModel.load("tmp/logistic-regression-model")
  //val model = CrossValidatorModel.load("tmp/logistic-regression-model")
  //val model = PipelineModel.load("tmp/decision-tree-model")
  val predictions = model.transform(pipeline.transform(test))

  predictions
    .coalesce(1)
    .select("auctionId", "prediction")
    .withColumn("isSold", ($"prediction" === 1.0).cast("Boolean"))
    .drop("prediction")
    .write.mode("overwrite").option("header", true).csv("tmp/submission.csv")


  println(predictions.toDF().count())
  println(test.count())

  spark.close()

}
