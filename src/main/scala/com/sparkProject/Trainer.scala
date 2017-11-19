package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,IDF}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.util.Calendar

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/
   val df: DataFrame = spark
     .read
     .option("header", true)  // Use first line of all files as header
     .option("inferSchema", "true") // Try to infer the data types of each column
     .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
      .parquet("./data/prepared_trainingset")//
    df.show(50)

    /** TF-IDF **/

      /* Tokenize text */

    val   tokenizer  =   new   RegexTokenizer()
      .setPattern( "\\W+" )
      .setGaps( true )
      .setInputCol( "text" )
      .setOutputCol( "tokens" )

    // check data tranformation
    val wordsData = tokenizer.transform(df)
    //wordsData.show(5)


    /** Remove Stop Words **/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // check data tranformation
    val filteredData = remover.transform(wordsData)
    //filteredData.show(1)


    /* fit a CountVectorizerModel from the corpus */
    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setMinDF(1)
      .fit(filteredData)

    // check data tranformation
    val featurizedData = cvModel.transform(filteredData)
    //featurizedData.show(1)


    /* TFIDF */
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("tfidf")

    val idfModel = idf.fit(featurizedData)

    // check data tranformation
    val rescaledData = idfModel.transform(featurizedData)
    //rescaledData.select("tfidf").show(5)

    /* Indexation of Country and Currency columns */

    val indexer1 = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .fit(rescaledData)
    val indexed1 = indexer1.transform(rescaledData)

    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .fit(rescaledData)

    // check data tranformation
    val indexedData = indexer2.transform(indexed1)
    //indexedData.show(5)

    /** VECTOR ASSEMBLER **/

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed",   "currency_indexed"))
      .setOutputCol("features")

    // check data tranformation
    //val output = assembler.transform(indexedData)
    //println("Assembled columns \"tfidf\", \"days_campaign\", \"hours_prepa\", \"goal\", \"country_indexed\",  " +
    //" \"currency_indexed\" to vector column 'features'")
   // output.select("features").show(5)


    /** LogisticRegression MODEL **/
    val   lr  = new  LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds( Array (0.7,0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE with all transformation and model stages**/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover,cvModel,idfModel,indexer1,indexer2, assembler,lr))





    /** TRAINING AND GRID-SEARCH **/

    // Split data into training (90%) and test (10%).
    val splits = df.randomSplit(Array(0.9, 0.1), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    // Fit the pipeline to training data.
    val model = pipeline.fit(training)

    // check pipline stages on test data.
    //model.transform(test).select("predictions", "raw_predictions").show(10)


    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of Regularization parameter values for the Logistic Model and minDF parametre for CounterVector
    // and determine best model using the evaluator.

    // For  Logistic regularization we will test the following values: 10e-8,   10e-6,   10e-4   et   10e-2
    // For CountVerctorizer we will test 55, 75, 95 minDF values
    // In each grid step we will use 70%   of training data and   30%  for validation (test).

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF, Array(55.0,75.0,95.0))
      .build()


    // A TrainValidationSplit requires an Evaluator.
    // we will use MulticlassClassificationEvaluator since it handle F1_score metric

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)


    // Run train validation split, and choose the best set of parameters.
    val best_model = trainValidationSplit.fit(training)

    // Make predictions on test data (10% of initial data). model used is the model with combination of parameters
    // that performed best.
    val df_WithPredictions = best_model.transform(test)

    val Test_f1Score = evaluator.evaluate(df_WithPredictions)

    println("F1 score on test data: " + Test_f1Score)

    df_WithPredictions.groupBy( "final_status" ,  "predictions").count.show()



    //Result of the execution
    //F1 score on test data: 0.6511234654139224
    //+------------+-----------+-----+
    //|final_status|predictions|count|
    //+------------+-----------+-----+
    //|           1|        0.0| 1279|
    // |           0|        1.0| 2267|
    //|           1|        1.0| 1914|
    //|           0|        0.0| 4428|
    //+------------+-----------+-----+

    //SAVE THE MODEL in ./data/Models directory

    val modelDir = "./data/Models/";
    val datestamp = Calendar.getInstance().getTime().toString.replaceAll(" ", ".").replaceAll(":", "_");
    val modelName = "LogisticRegression__"
    val filename = modelDir.concat(modelName).concat(datestamp)
    model.save(filename);


  }
}
