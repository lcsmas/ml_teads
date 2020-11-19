import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{hour, minute, to_timestamp}

object GeneratePipeline extends App{
  import org.apache.spark.sql.SparkSession;
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.OneHotEncoder
  import org.apache.spark.sql.functions.countDistinct

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  def transform_train_data(df: DataFrame): DataFrame = {
    df
      .withColumn("hour", hour(to_timestamp($"timeStamp".cast("Int")))).drop("timeStamp")
      .withColumn("isSoldInt", ($"isSold" === "true").cast("Int")).drop("isSold")
  }

  val full_train_data = spark.read
    .format("csv")
    .option("header", "true")
    .csv("data/train.csv")
    .toDF()


  val Array(train_df, test_df) = full_train_data
    .transform(transform_train_data)
    .randomSplit(Array(.8, .2), seed=42)
  train_df.write.save("tmp/training_set")
  test_df.write.save("tmp/test_set")

//  val train_df = spark.read
//    .format("csv")
//    .option("header", "true")
//    .csv("data/train.csv")
//    .toDF()
//    .transform(transform_timestamp)


//  val test_df = spark.read
//    .format("csv")
//    .option("header", "true")
//    .csv("data/test.csv")
//    .toDF()
//    .transform(transform_timestamp)

  val categoricalCols = train_df.dtypes.filter{case (field, dataType) => dataType == "StringType" && field != "auctionId"}.map(_._1)
  val indexOutputCols = categoricalCols.map(_ + "Index")
  val oheOutputCols = categoricalCols.map(_ + "OHE")

  val stringIndexer = new StringIndexer()
    .setInputCols(categoricalCols)
    .setOutputCols(indexOutputCols)
    .setHandleInvalid("keep")

  val oheEncoder = new OneHotEncoder()
    .setInputCols(indexOutputCols)
    .setOutputCols(oheOutputCols)

  val vecAssembler = new VectorAssembler()
    .setInputCols(oheOutputCols)
    .setOutputCol("features")

  val logisticRegression = new LogisticRegression()
    .setLabelCol("isSoldInt")
    .setFeaturesCol("features")
    .setMaxIter(50)
    .setRegParam(0.03)

  val pipeline = new Pipeline()
    .setStages(Array(stringIndexer, oheEncoder, vecAssembler, logisticRegression))

  val model = pipeline.fit(train_df)
  model.write.overwrite().save("tmp/logistic-regression-model")

  val predictions = model.transform(test_df)

  var evaluator_binary = new BinaryClassificationEvaluator()
    .setLabelCol("isSoldInt")
    .setRawPredictionCol(logisticRegression.getRawPredictionCol)

  var m_areaUnderROC = evaluator_binary.evaluate(predictions)
  var metricname = evaluator_binary.getMetricName

  println(f"$metricname is $m_areaUnderROC%.3f")

}
