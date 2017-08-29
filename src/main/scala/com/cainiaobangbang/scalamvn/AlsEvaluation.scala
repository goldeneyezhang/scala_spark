package com.cainiaobangbang.scalamvn
import java.io.File
import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS,Rating,MatrixFactorizationModel}
import org.joda.time.format._
import org.joda.time._
import org.joda.time.Duration
import org.jfree.data.category.DefaultCategoryDataset
import org.apache.spark.mllib.regression.LabeledPoint
object AlsEvaluation {
  def main(args:Array[String]){
   //set not show surplus information
   SetLogger
   println("=====prepare stage=====")
   val(trainData,validationData,testData)=PrepareData()
    trainData.persist()
   validationData.persist()
   testData.persist()
   println("=====train validation stage=====")
  val bestModel=trainValidation(trainData,validationData)
  println("===test stage=====")
   val testRmse=computeRMSE(bestModel,testData)
   println("use testData test bestModel,"+" result rmse="+testRmse)
   trainData.unpersist()
   validationData.unpersist()
   testData.unpersist()
  }
   def SetLogger={
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF)
  }
  
  def PrepareData():(RDD[Rating],RDD[Rating],RDD[Rating])={
    //1.create user rating data
    val sc=new SparkContext(new SparkConf().setAppName("Recommend").setMaster("local[4]"))
    val DataDir="data"
    println("start read user score")
    val rawUserData=sc.textFile("file:/home/hadoop/workspace/scalamvn/data/u.data")
    val rawRatings=rawUserData.map(_.split("\t").take(3))
    val ratingsRDD=rawRatings.map{case Array(user,movie,rating)=>Rating(user.toInt,movie.toInt,rating.toDouble)}
    println("total "+ratingsRDD.count.toString()+" ratings")
    //2.create movie id and name map table
    println("start read movie data")
    val itemRDD=sc.textFile(new File(DataDir,"u.item").toString)
    val movieTitle=itemRDD.map(line=>line.split("\\|").take(2)).map(array=>(array(0).toInt,array(1))).collect().toMap
    //3.show data record count
    val numRatings=ratingsRDD.count()
    val numUsers=ratingsRDD.map(_.user).distinct().count()
    val numMovies=ratingsRDD.map(_.product).distinct().count()
    println("total:ratings:"+numRatings+" User "+numUsers+" Movie "+numMovies)
    //4.random split data to 3 parts
    println("split data")
    val Array(trainData,validationData,testData)=ratingsRDD.randomSplit(Array(0.8,0.1,0.1))
    println(" trainData:"+trainData.count()+ " validationData:"+validationData.count()+" testData:"+testData.count())
    return (trainData,validationData,testData)
  }
  def trainValidation(trainData:RDD[Rating],validationData:RDD[Rating]):MatrixFactorizationModel={
    println("-----evalation rank parameters use-----")
    evaluateParameter(trainData,validationData,"rank",Array(5,10,15,20,50,100),Array(10),Array(0.1))
   println("-----evalation numIterations-----")
   evaluateParameter(trainData,validationData,"numIterations",Array(10),Array(5,10,15,20,25),Array(0.1))
   println("-----evaluation lamda-----")
   evaluateParameter(trainData,validationData,"lambda",Array(10),Array(10),Array(0.05,0.1,1,5,10.0))
   println("-----all parameters cross evaluate best parameters combination-----")
   val bestModel=evaluateAllParameter(trainData,validationData,Array(5,10,15,20,25),Array(5,10,15,20,25),Array(0.05,0.1,1,5,10.0))
   return bestModel
  }
  def evaluateParameter(trainData:RDD[Rating],validationData:RDD[Rating],evaluateParameter:String,rankArray:Array[Int],numIterationsArray:Array[Int],lambdaArray:Array[Double])={
    var dataBarChart=new DefaultCategoryDataset()
    var dataLineChart=new DefaultCategoryDataset()
    for(rank<-rankArray;numIterations<-numIterationsArray;lambda<-lambdaArray){
      val (rmse,time)=trainModel(trainData,validationData,rank,numIterations,lambda)
      val parameterData=evaluateParameter match{
        case "rank"=>rank;
        case "numIterations"=>numIterations;
        case "lambda"=>lambda
      }
      dataBarChart.addValue(rmse,evaluateParameter,parameterData.toString())
      dataLineChart.addValue(time, "Time", parameterData.toString())
      
    }
    Chart.plotBarLineChart("ALS evaluations  "+evaluateParameter,evaluateParameter,"RMSE",0.58,5,"Time",dataBarChart,dataLineChart)
  }
  def trainModel(trainData:RDD[Rating],validationData:RDD[Rating],rank:Int,iterations:Int,lambda:Double):(Double,Double)={
    val startTime=new DateTime()
    val model=ALS.train(trainData,rank,iterations,lambda)
    val endTime=new DateTime()
    val Rmse=computeRMSE(model,validationData)
    val duration=new Duration(startTime,endTime)
    println(f"train parameters: rank:$rank%3d,iterations:$iterations%.2f,lambda=$lambda%.2f result Rmse=$Rmse%.2f"+"train need time"+duration.getMillis+"millisecond")
    (Rmse,duration.getStandardSeconds)
  }
  def computeRMSE(model:MatrixFactorizationModel,RatingRDD:RDD[Rating]):Double={
    val num=RatingRDD.count()
    val predictedRDD=model.predict(RatingRDD.map(r=>(r.user,r.product)))
    val predictedAndRatings=predictedRDD.map(p=>((p.user,p.product),p.rating)).join(RatingRDD.map(r=>((r.user,r.product),r.rating))).values
    math.sqrt(predictedAndRatings.map(x=>(x._1-x._2)*(x._1-x._2)).reduce(_+_)/num)
  }
  def evaluateAllParameter(trainData:RDD[Rating],validationData:RDD[Rating],rankArray:Array[Int],numIterationsArray:Array[Int],lambdaArray:Array[Double]):MatrixFactorizationModel={
    val evaluations=
      for(rank<-rankArray;numIterations<-numIterationsArray;lambda<-lambdaArray)
        yield{
         val (rmse,time)=trainModel(trainData,validationData,rank,numIterations,lambda)
         (rank,numIterations,lambda,rmse)
      }
    val Eval=(evaluations.sortBy(_._4))
    val BestEval=Eval(0)
    println("best model parameters: rank:"+BestEval._1+",iterations:"+BestEval._2+"lambda"+BestEval._3+",result rmse="+BestEval._4)
    val bestModel=ALS.train(trainData,BestEval._1,BestEval._2,BestEval._3)
    (bestModel)
  }
}