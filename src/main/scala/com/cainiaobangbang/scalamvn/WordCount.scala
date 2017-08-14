package com.cainiaobangbang.scalamvn
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.rdd.RDD
object WordCount {
  def main(args:Array[String]):Unit={
    Logger.getLogger("org").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress","false")
    println("start runwordcount")
    val sc=new SparkContext(new SparkConf().setAppName("wordCount").setMaster("local[4]"))
    println("start read text")
    val textFile=sc.textFile("data/LICENSE.txt")
    println("start create rdd")
    val countsRDD=textFile.flatMap(line=>line.split(" ")).map(word=>(word,1)).reduceByKey(_+_)
    println("start save text")
    try{
      countsRDD.saveAsTextFile("data/output")
      println("save success")
    }catch{
      case e:Exception=>println("directory exists,remove old directory")
    }
  }
}