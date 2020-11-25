import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{avg, dayofmonth, dayofweek, dayofyear, hour, minute, month, to_timestamp, weekofyear, when, year}

object GenerateTransformedDF extends App {
  import org.apache.spark.sql.SparkSession;
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.OneHotEncoder

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

  val train = spark.read
    .format("csv")
    .option("header", "true")
    .csv("data/train.csv")
    .toDF()
    .transform(withTimeInfo)

//  val Array(avgDF, rest) = fullTrain.randomSplit(Array(0.1, 0.9))
//  val aggregatedByPlacementId = avgDF.groupBy($"placementId")
//    .agg(avg($"isSoldInt").as("averageHasSoldByPlacementId"))
//  val aggByCountry = avgDF.groupBy($"country")
//    .agg(avg($"isSoldInt").as("averageHasSoldByCountry"))

//  val train = rest.join(aggregatedByPlacementId, Seq("placementId"),"left")
//
  //val train = trainAndPlacementAgg.join(aggByCountry,Seq("country"),"left")


//  val categoricalFeatures = train.select(
//    "hour",
//    "dayOfWeek",
//    "dayOfMonth",
//    "dayOfYear",
//    "month",
//    "year",
//    "weekOfYear",
//    "websiteId",
//    "placementId",
//    "device",
//    "articleSafenessCategorization",
//    "opeartingSystem",
//    "hashedRefererDeepThree",
//    "country",
//    "browser",
//    "browserVersion",
//    "environmentType",
//    "integrationType"
//  )

  val categoricalCols = train.dtypes.filter{
    case (field, dataType) => dataType == "StringType" && field != "auctionId"
  }.map(_._1)

  val indexOutputCols = categoricalCols.map(_ +"Index")
  val oheOutputCols = categoricalCols.map(_ + "OHE")

  val stringIndexer = new StringIndexer()
    .setInputCols(categoricalCols)
    .setOutputCols(indexOutputCols)
    .setHandleInvalid("keep")

  val oheEncoder = new OneHotEncoder()
    .setInputCols(indexOutputCols)
    .setOutputCols(oheOutputCols)

  val numericCols = train.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" }.map(_._1)

  val assemblerInputs = oheOutputCols ++ numericCols

  val colsToImpute = Array()

//  val imputer = new Imputer("")
//    .setStrategy("median")
//    .setInputCols(colsToImpute)
//    .setOutputCols(colsToImpute)

  val vecAssembler = new VectorAssembler()
    .setInputCols(assemblerInputs)
    .setOutputCol("features")

  val pipeline = new Pipeline()
    .setStages(Array(stringIndexer, oheEncoder, vecAssembler))

  pipeline.fit(train).write.overwrite().save("tmp/fitted-pipeline")
  pipeline.fit(train).transform(train).write.mode("overwrite").save("tmp/transformed-train")

  spark.close()

}
