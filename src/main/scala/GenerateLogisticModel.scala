import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{min, max}

object GenerateLogisticModel extends App{
  import org.apache.spark.sql.SparkSession;
  import org.apache.spark.ml.classification.LogisticRegression

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val logisticRegression = new LogisticRegression()
    .setLabelCol("isSoldInt")
    .setFeaturesCol("features")
    .setRegParam(0.01)
    .setMaxIter(50)
    .setElasticNetParam(0.5)
    .setThreshold(0.3888825619130765)


  val paramGrid = new ParamGridBuilder()
    .addGrid(logisticRegression.regParam, Array(0.0,0.001))
    .build()

  val evaluator = new BinaryClassificationEvaluator()
    .setRawPredictionCol(logisticRegression.getRawPredictionCol)
    .setLabelCol("isSoldInt")

  val cv = new CrossValidator()
    .setEstimator(logisticRegression)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(2)

  val train = spark.read.load("tmp/transformed-train")
  val cvModel = logisticRegression.fit(train)
  cvModel.write.overwrite().save("tmp/logistic-regression-model")

//  println(cvModel.bestModel.params.mkString("\n"))
//  println(cvModel.avgMetrics.mkString("\n"))

  showMaxFMeasureWithThreshold

  val showMaxFMeasureWithThreshold : Unit = {
    val fMeasure = cvModel.binarySummary.fMeasureByThreshold
    val areaUnderROC = cvModel.binarySummary.areaUnderROC
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThresholdForFMeasure = fMeasure.where($"F-Measure" === maxFMeasure)
      .select("threshold").head().getDouble(0)
    println(s"Max F-Measure : $maxFMeasure for threshold : $bestThresholdForFMeasure")
    println(s"Area under ROC : $areaUnderROC)")
  }
  spark.close()

}
