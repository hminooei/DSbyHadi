%md

### Logistic Regression Classification in Scala using Spark ML

*Hadi*

%md

#### Spark ML:
  works with DataFrames (vs. Spark Mllib works with RDDs)

%md

### Concepts

<b>DataFrame</b>: This ML API uses DataFrame from Spark SQL as an ML dataset, which can hold a variety of data types. E.g., a DataFrame could have different columns storing text, feature vectors, true labels, and predictions.

<b>Transformer</b>: A Transformer is an algorithm which can transform one DataFrame into another DataFrame. E.g., an ML model is a Transformer which transforms a DataFrame with features into a DataFrame with predictions.

<b>Estimator</b>: An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.

<b>Pipeline</b>: A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.

(from spark.apache.org)

import org.apache.spark.sql.DataFrame

val data = spark.read.option("header", "true").csv("/Users/hadi.minooei/Documents/DSbyHadi/car_ownership.csv")

data.count
data.columns

data.printSchema

// Change income type from String to Double

import org.apache.spark.sql.types.DoubleType

val dataDF = data
  .withColumn("income_double", data("income").cast(DoubleType))
  .drop("income")
  .withColumnRenamed("income_double", "income")

dataDF.printSchema

dataDF.take(3)

// We need to predict owns_car --> response, outcome or dependent variable
// Three features to be used: state, gender and income
// 'income' is a continuous feature
// 'state' and 'gender' are categorical features

// We need to index the owns_car i.e. true -> 0.0 and false -> 1.0 or vice versa.
// For this we use StringIndexer.

import org.apache.spark.ml.feature.{StringIndexer}

val labelIndexer = new StringIndexer()
  .setInputCol("owns_car")
  .setOutputCol("owns_car_index")
  .fit(dataDF)

labelIndexer.transform(dataDF).columns
labelIndexer.transform(dataDF).take(4)

// Three features to be used: state, gender and income
// 'income' is a continuous feature
// 'state' and 'gender' are categorical features

%md

<b>One-hot encoding</b> maps a column of label indices to a column of binary vectors, with at most a single one-value.
  This encoding allows algorithms which expect continuous features, such as Logistic Regression, to use categorical features.

// Illustration
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val df = spark.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .fit(df)
val indexed = indexer.transform(df)

val encoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVec")

val encoded = encoder.transform(indexed)
encoded.show()

// Two dummy binary variables generated: (categoryA, categoryC)
// This is like defining two binary variables categoryA, categoryC but in the short vector format (it's a
// dense vector)
// a: (2,[0],[1.0]) -> vector of len 2, with value at entry 0 equals 1.0 i.e. (1.0, 0.0)
// c: (2,[1],[1.0]) -> vector of len 2, with value at entry 1 equals 1.0 i.e. (0.0, 1.0)
// b: (2,[],[]) -> vector of len 2, with all etries 0.0 i.e. (0.0, 0.0)


import org.apache.spark.ml.feature.OneHotEncoder;

val genderIndexer = new StringIndexer()
  .setInputCol("gender")
  .setOutputCol("gender" + "_index")
  .setHandleInvalid("skip")
  .fit(dataDF)

val genderOneHotEncoder = new OneHotEncoder()
  .setInputCol("gender" + "_index")
  .setOutputCol("gender" + "_vec")

val stateIndexer = new StringIndexer()
  .setInputCol("state")
  .setOutputCol("state" + "_index")
  .setHandleInvalid("skip")
  .fit(dataDF)

val stateOneHotEncoder = new OneHotEncoder()
  .setInputCol("state" + "_index")
  .setOutputCol("state" + "_vec")

val test1 = stateIndexer.transform(dataDF)
val test2 = stateOneHotEncoder.transform(test1)

test1.columns
test1.take(6)

test2.columns
test2.take(6)

val test3 = genderIndexer.transform(test2)
val test4 = genderOneHotEncoder.transform(test3)

test4.columns
test4.head

// Putting all featurs together in a vector

// features: stateCA, stateTX, genderMale, income

import org.apache.spark.ml.feature.VectorAssembler

val featuresAssembler = new VectorAssembler()
  .setInputCols(Array("income", "state_vec", "gender_vec"))
  .setOutputCol("features")

val assembled = featuresAssembler.transform(test4)
assembled.columns
assembled.head

%md

Normalizing the continuous feature:

  MinMaxScaler transforms a dataset of Vector rows, rescaling each feature to a specific range (often [0, 1]). It takes parameters:

  min: 0.0 by default. Lower bound after transformation, shared by all features.
  max: 1.0 by default. Upper bound after transformation, shared by all features.

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors

val scaler = new MinMaxScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// First fit to data to compute summary statistics
val scalerModel = scaler.fit(assembled)

// Then transform to rescale each feature to [0, 1] range
val scaledData = scalerModel.transform(assembled)

scaledData.columns
scaledData.select("features", "scaledFeatures").show()

// Define the logistic regression model; need to set the outcome column

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}

val lr = new LogisticRegression()
  .setLabelCol("owns_car_index")
  .setFeaturesCol("scaledFeatures")
  .setMaxIter(20)
  .setRegParam(0.05)
  .setElasticNetParam(0.0) // 0.0 is l2 regularization, 1.0 is l1 regularization

// Now we put the pipeline together

import org.apache.spark.ml.{Pipeline, PipelineModel}

val pipeline = new Pipeline().setStages(Array(genderIndexer, genderOneHotEncoder, stateIndexer, stateOneHotEncoder) ++ Array(featuresAssembler, scaler, labelIndexer, lr))

// We do train-test split first.

val Array(trainDF, testDF) = dataDF.randomSplit(Array(0.65, 0.35))

trainDF.count
trainDF.take(3)

testDF.count
testDF.take(3)

val model = pipeline.fit(trainDF)

// We can now make predictions using 'model'

val testPreds = model.transform(testDF)

testPreds.columns
testPreds.head

labelIndexer.labels

// so true mapped to index 0.0
// false mapped to index 1.0

testPreds.select("owns_car", "probability", "prediction").take(3)

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}

val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("owns_car_index")
  .setRawPredictionCol("probability")
  .setMetricName("areaUnderROC") //"areaUnderROC" (default), or "areaUnderPR"

evaluator.evaluate(testPreds)

val trainPreds = model.transform(trainDF)
trainPreds.head
evaluator.evaluate(trainPreds)

import org.apache.spark.ml.linalg.DenseVector

val scoreAndLabelsTrain = trainPreds.select("probability", "owns_car_index")
  .rdd
  .map(row =>
    (row.getAs[DenseVector]("probability")(1), row.getAs[Double]("owns_car_index"))
  )

val scoreAndLabelsTest = testPreds.select("owns_car_index", "probability")
  .rdd
  .map(row =>
    (row.getAs[DenseVector]("probability")(1), row.getAs[Double]("owns_car_index"))
  )

// This approach gives more metrics other than auc roc and aur pr, e.g. precision, recall, ...
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val metricsTrain = new BinaryClassificationMetrics(scoreAndLabelsTrain)
val metricsTest = new BinaryClassificationMetrics(scoreAndLabelsTest)

metricsTrain.areaUnderROC
metricsTest.areaUnderROC

metricsTrain.roc.take(5)
metricsTest.roc.take(5)

val f1Scores = metricsTrain.fMeasureByThreshold
f1Scores.collect.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}

%md #### Putting everything together

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer, VectorAssembler, OneHotEncoder};
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}


val IndexerPostfix = "_index"
val EncoderPostfix = "_vec"
val FeaturesVectorColumn = "features"
val ScaledFeaturesVectorColumn = "features_scaled"
val ActionForInvalidFlag = "skip" // Can also be "keep" or "error"
val TrainTestSplitRatio = 0.65

def trainLRModel(
    allData: DataFrame,
    labelCol: String,
    continuousFeatures: Array[String],
    categoricalFeatures: Array[String]): Model[_] = {

  val labelIndexer = new StringIndexer()
    .setInputCol(labelCol)
    .setOutputCol(labelCol + IndexerPostfix)
    .fit(allData)

  // Encoding categorical features
  val toOneHotEncoder = (featureName: String) => Seq(
    new StringIndexer()
      .setInputCol(featureName)
      .setOutputCol(featureName + IndexerPostfix)
      .setHandleInvalid(ActionForInvalidFlag)
      .fit(allData),
    new OneHotEncoder()
      .setInputCol(featureName + IndexerPostfix)
      .setOutputCol(featureName + EncoderPostfix))

  val categoricalStages = categoricalFeatures.flatMap(toOneHotEncoder)

  val featuresNames = continuousFeatures ++ categoricalFeatures.map(_ + EncoderPostfix)

  val featuresAssembler = new VectorAssembler()
    .setInputCols(featuresNames.toArray)
    .setOutputCol(FeaturesVectorColumn)

  val scaler = new MinMaxScaler()
    .setInputCol(FeaturesVectorColumn)
    .setOutputCol(ScaledFeaturesVectorColumn)

  val lr = new LogisticRegression()
    .setLabelCol(labelCol + IndexerPostfix)
    .setFeaturesCol(ScaledFeaturesVectorColumn)
    .setMaxIter(20)
    .setRegParam(0.05)
    .setElasticNetParam(0.0)

  val pipeline = new Pipeline().setStages(categoricalStages ++ Array(featuresAssembler, scaler, labelIndexer, lr))

  val Array(trainDF, testDF) = allData.randomSplit(Array(TrainTestSplitRatio, 1 - TrainTestSplitRatio))

  val trainedModel = pipeline.fit(trainDF)

  // do other things like evaluation..

  trainedModel
}

dataDF.columns

val trainedModel = trainLRModel(dataDF, "owns_car", Array("income"), Array("state", "gender"))

val preds = trainedModel.transform(dataDF)
preds.columns
preds.head







