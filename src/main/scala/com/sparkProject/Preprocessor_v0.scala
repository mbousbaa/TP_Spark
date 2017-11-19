package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object PreprocessorOld {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    /** *****************************************************************************
      *
      * TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    import spark.implicits._

    val sc = spark.sparkContext


    /** 1 - CHARGEMENT DES DONNEES */

    /*val train_df_original = spark.read.text("/Users/maha/Documents/myProjects/TP_ParisTech_2017_2018_starter/spark-warehouse/train_3.csv")
    train_df_original.show()

    val train_df_cleaned = train_df_original.withColumn("replaced", regexp_replace($"value","\"{2,}",""))
    train_df_cleaned.show()
    train_df_cleaned.select("replaced").write.csv("/Users/maha/Documents/myProjects/TP_ParisTech_2017_2018_starter/spark-warehouse/train_cleaned.csv")

*/


    val train_df = spark.read
      .format("com.databricks.spark.csv")
      .option("header", "true") //reading the headers
     // .option("mode", "DROPMALFORMED")
      /*.option("quoteMode", "ALL")*/
     //.option("inferschema","true")
      .option("mode", "PERMISSIVE")
      .option("nullValue","false")

      //.option("escape", "\"")
      //.option("inferSchema", "true")

      .option("parserLib", "univocity")
      //.option("charset", "UTF-8")
      .load("/Users/maha/Documents/myProjects/TP_ParisTech_2017_2018_starter/spark-warehouse/train_3.csv").
      toDF("project_id", "name", "desc", "goal", "keywords", "disable_communication", "country", "currency", "deadline",
        "state_changed_at", "created_at", "launched_at", "backers_count", "final_status")

    train_df.show()
    println("Nb of row : " + train_df.count())
    println("Nb of colomun  : " + train_df.columns.length)

    train_df.printSchema()


    /*[ ("project_id","string"),
            ("name", "string"),
            ("desc", "string"),
            ("goal","int"),
            ("keyword","string"),
            ("disable_communication", "string"),
            ("country", "string"),
            ("currency","string"),
            ("deadline","string"),
            ("state_changed_at", "string"),
            ("created_at","string"),
            ("launched_at","string"),
            ("backers_count","int"),
            ("final_status","string")]

            ]*/


    /*val toInt = udf[Int, String](_.toInt)
    val toDouble = udf[Double, String](_.toDouble)*/

    /* Conversion Type Goal en long*/


    val df2 = train_df.withColumnRenamed("goal", "goal_old").withColumnRenamed("backers_count", "backers_count_old")


    val train_df_T = df2.withColumn("goal", df2.col("goal_old").cast("long")).drop("goal_old")
          .withColumn("backers_count", df2.col("backers_count_old").cast("int")).drop("backers_count_old")

    train_df_T.printSchema()



       /* df.filter(df(colName).isNull || df(colName) === "" || df(colName).isNaN).count() */

       // train_df_T.printSchema()


        /** 2 - CLEANING **/
    /*groupements de data */
        train_df_T.groupBy("disabled_communication").count.orderBy($"count".desc).show()


    /** Suppression des exemples dont "goal" < 0 */


        println(" nb de ligne avec goal <0:" + train_df_T.filter($"goal" < 0).count())
        println(" nb de ligne avec backers_count <0:" + train_df_T.filter($"backers_count" < 0).count())
        train_df_T.describe("goal", "backers_count").show()

       //train_df_T.filter($"project_id".isNull ||  $"desc".isNull ||  $"keywords".isNull ||  $"country".isNull || $"currency".isNull)

        println ("Nb of Final status different from 0 and 1 " +train_df_T.filter(($"final_status" =!=0 && $"final_status" =!=1)).count())

       // val train_df_2 = train_df_T.filter(($"final_status" =!=0 && $"final_status" =!=1)).drop("final_status")
        val train_df_tmp = train_df_T.withColumn("final_status", when($"final_status" =!=0 && $"final_status" =!=1, null))
        val train_df_2 = train_df_tmp.drop("final_status")
        println ("Nb of Final status different from 0 and 1 " +train_df_2.filter(($"final_status" =!=0 && $"final_status" =!=1)).count())



        println(" Nb of project_id null :" + train_df_2.filter("project_id is null").count())
        train_df_2.describe("project_id","name","country","backers_count").show()

    /* Cleaning Project_id  */


        train_df_2.select("project_id").show()

        println(" Nb of project_id does not respect pattern kkst :" + train_df_2.filter(not($"project_id" rlike "kkst*")).count())

        val train_df_tmp2 = train_df_2.withColumn("project_id",when(not($"project_id" rlike "kkst*"),null))
        val train_df_3 = train_df_tmp2.drop("project_id")
        println(" Nb of project_id does not respect pattern kkst :" + train_df_3.filter(not($"project_id" rlike "kkst*")).count())

        train_df_2.describe("project_id","name","country","backers_count").show()

        train_df_3.describe("project_id","name","country","backers_count").show()







    /*train_df_T.groupBy("disable_communication").count().show()

    train_df_T.drop($"state_changed_at").drop($"backers_count").drop($"disable_communication") */

      //  train_df_T.filter($"country".isNull).show()
        //groupBy("currency").count().orderBy($"count".desc).show(50)*/





    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/


  }

}
