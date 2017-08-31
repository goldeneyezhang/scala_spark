package com.cainiaobangbang.scalamvn
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.joda.time._
import org.jfree.data.category.DefaultCategoryDataset
object RunDecisionTreeBinary {
  def main(args:Array[String]):Unit={
    SetLogger
    val sc=new SparkContext(new SparkConf().setAppName("DecisionTreeBinary").setMaster("local[4]"))
    println("RunDecisionTreeBinary")
    println("=======data prepare stage=====")
    val(trainData,validationData,testData,categoriesMap)=PrepareData(sc)
    trainData.persist()
    validationData.persist()
    testData.persist()
    println("=====train evaluation stage=====")
    val model=trainEvaluate(trainData,validationData)
    println("=====test stage=====")
    val auc=evaluateModel(model,testData)
    println("use testata test best model,result auc:"+auc)
    println("=====predict data=====")
    PredictData(sc.model,categoriesMap)
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
   
}