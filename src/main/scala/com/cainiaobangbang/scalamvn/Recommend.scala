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
import scala.collection.immutable.Map

object Recommend {
   def main(args:Array[String]):Unit={
     SetLogger
    val (ratings,movieTitle)=PrepareData()
    val model=ALS.train(ratings,10,10,0.1)
     recommend(model,movieTitle)
   }
  def recommend(model:MatrixFactorizationModel,movieTitle:Map[Int,String])={
    var choose="1"
    while(choose!="3"){//if 3,leave,stop running
      println("please choose 1. input userid,recommend movie 2.input movieid,recommend user 3.leave")
      choose=readLine()
      if(choose=="1"){
        println("please input userid")
        val inputUserID=readLine()
        RecommendMovies(model,movieTitle,inputUserID.toInt)
      }
      else if(choose=="2"){
        println("please input movieid")
        val inputMovieID=readLine()
        RecommendUsers(model,movieTitle,inputMovieID.toInt)
      }
    }
  }
    def SetLogger={
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF)
  }
    def PrepareData():(RDD[Rating],Map[Int,String])={
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
    return (ratingsRDD,movieTitle)
  }
  def RecommendMovies(model:MatrixFactorizationModel,movieTitle:Map[Int,String],inputUserID:Int)={
    val RecommendMovie=model.recommendProducts(inputUserID,10)
    var i=1
    println("userid="+inputUserID+"recommend movies")
    RecommendMovie.foreach{r=>println(i.toString()+"."+movieTitle(r.product)+"score:"+r.rating.toString())
     i+=1
  }
  }
  def RecommendUsers(model:MatrixFactorizationModel,movieTitle:Map[Int,String],inputMovieID:Int)={
    val RecommendUser=model.recommendUsers(inputMovieID, 10)//inputMovieID recommend ten users
    var i=1
    println("movieid="+inputMovieID+"moviename"+movieTitle(inputMovieID.toInt)+"userids=")
    RecommendUser.foreach{r=>println(i.toString+"userid:"+r.user+" score:"+r.rating)
     i+=1
    }
  }
}