import org.apache.spark.ml.PipelineModel

object GenerateCsvSubmission extends App{
  import org.apache.spark.sql.DataFrame
  import org.apache.spark.sql.functions.{hour, minute, to_timestamp}
  import org.apache.spark.sql.SparkSession;

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  def transform_test_data(df: DataFrame): DataFrame = {
    df
      .withColumn("hour", hour(to_timestamp($"timeStamp".cast("Int")))).drop("timeStamp")
  }

  val test_df = spark.read
    .format("csv")
    .option("header", "true")
    .csv("data/test.csv")
    .toDF()
    .transform(transform_test_data)

  val model = PipelineModel.load("tmp/logistic-regression-model")
  val predictions = model.transform(test_df)

  predictions
    .coalesce(1)
    .select("auctionId", "prediction")
    .withColumn("isSold", ($"prediction" === 1.0).cast("Boolean"))
    .drop("prediction")
    .write.mode("overwrite").option("header", true).csv("tmp/submission.csv")

  println(predictions.toDF().count())
  println(test_df.count())


}
