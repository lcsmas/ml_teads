object Main extends App{
  import org.apache.spark.sql.SparkSession;

  val spark = SparkSession.builder()
    .master("local[*]")
    .getOrCreate();

  val train_df = spark.read
    .format("csv")
    .option("header", "true")
    .csv("data/train.csv")


  train_df.select("auctionId").show()

}
