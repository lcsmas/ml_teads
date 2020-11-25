import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{hour, minute, to_timestamp}

object GenerateTreePipeline extends App {
  import org.apache.spark.sql.SparkSession;
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.feature.OneHotEncoder
  import org.apache.spark.sql.functions.countDistinct

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  // Train a RandomForest model.
  val rf = new DecisionTreeClassifier()
    .setLabelCol("isSoldInt")
    .setFeaturesCol("features")
    .setMaxDepth(0)
    //.setNumTrees(50)


  val train = spark.read.load("tmp/transformed-train")
  val model = rf.fit(train)

  model.write.overwrite().save("tmp/decision-tree-model")

  spark.close()
}
