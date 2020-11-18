import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{hour, to_timestamp}

object ThresholdTuning extends App {
  import org.apache.spark.sql.SparkSession;
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.OneHotEncoder

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  def transform_data(df: DataFrame): DataFrame = {
    df
      .withColumn("hour", hour(to_timestamp($"timeStamp".cast("Int")))).drop("timeStamp")
      .withColumn("isSoldInt", ($"isSold" === "true").cast("Int")).drop("isSold")
  }

  val data =  spark.read
    .format("csv")
    .option("header", "true")
    .csv("data/train.csv")
    .toDF()
    .transform(transform_data)

  // Split data into training (60%) and test (40%)
  val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
  training.cache()

  val categoricalCols = training.dtypes.filter{case (field, dataType) => dataType == "StringType" && field != "auctionId"}.map(_._1)
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

  val pipeline = new Pipeline()
    //.setStages(Array(stringIndexer, oheEncoder, vecAssembler, logisticRegression))
    .setStages(Array(stringIndexer, oheEncoder, vecAssembler))

  val pipeLineModel = pipeline.fit(training)
  val pipeLinedTraining = pipeLineModel.transform(training)

  val logisticRegression = new LogisticRegression()
    .setLabelCol("isSoldInt")
    .setFeaturesCol("features")
    .setMaxIter(500)
    .setRegParam(0.03)

  val model = logisticRegression.fit(pipeLinedTraining)

  val trainingSummary = model.binarySummary

  trainingSummary.fMeasureByThreshold.show(50)
  //println(s"fMeasureByThreshold: ${trainingSummary.fMeasureByThreshold}")

}
