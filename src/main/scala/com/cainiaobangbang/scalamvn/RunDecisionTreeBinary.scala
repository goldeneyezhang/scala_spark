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
    println()
    println("是否需要进行参数调试(Y,N)?")
    if(readLine()=="Y")
    {
      val model=parametersTunning(trainData,validationData)
      println("=====测试阶段=====")
      val auc=evaluateModel(model,testData)
      println("使用testata测试最佳模型，结果AUC:"+auc)
      println("=====预测数据=====")
      PredictData(sc,model,categoriesMap)
    }
    else
    {
    val model=trainEvaluate(trainData,validationData)
    println("=====test stage=====")
    val auc=evaluateModel(model,testData)
    println("use testata test best model,result auc:"+auc)
    println("=====predict data=====")
    PredictData(sc,model,categoriesMap)
    trainData.unpersist()
    validationData.unpersist()
    testData.unpersist()
    }
  }
   def SetLogger={
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF)
  }
   def PrepareData(sc:SparkContext):(RDD[LabeledPoint],RDD[LabeledPoint],RDD[LabeledPoint],Map[String,Int])={
     //1、导入转换数据
     println("开始导入数据...")
     val rawDataWithHeader=sc.textFile("data/train.tsv")
     //删除第一行表头
     val rawData=rawDataWithHeader.mapPartitionsWithIndex{(idx,iter)=>if(idx==0) iter.drop(1) else iter}
     val lines=rawData.map(_.split("\t"))
     //2、创建训练评估所需的数据RDD[LabeledPoint] 字段0～2忽略 字段3是分类特征字段 字段4～25是数值特征字段 字段26label标签字段
     val categoriesMap=lines.map(fields=>fields(3)).distinct.collect.zipWithIndex.toMap
     val labelpointRDD=lines.map{fields=>
       val trFields=fields.map(_.replaceAll("\"",""))
       val categoryFeaturesArray=Array.ofDim[Double](categoriesMap.size)
       val categoryIdx=categoriesMap(fields(3))
       categoryFeaturesArray(categoryIdx)=1
       val numericalFeatures=trFields.slice(4,fields.size-1)
       .map(d=>if(d=="?")0.0 else d.toDouble)
       val label=trFields(fields.size-1).toInt
       LabeledPoint(label,Vectors.dense(categoryFeaturesArray++numericalFeatures))
     }
     //3.以随机分割数据分为3部分并返回
     val Array(trainData,validationData,testData)=labelpointRDD.randomSplit(Array(0.8,0.1,0.1))
     println("将数据分trainData:"+trainData.count()+"  validationData:"+validationData.count()+"  testData:"+testData.count())
     return (trainData,validationData,testData,categoriesMap)//返回数据
   }
   def trainEvaluate(trainData:RDD[LabeledPoint],validationData:RDD[LabeledPoint]):DecisionTreeModel={
     println("开始训练")
     val(model,time)=trainModel(trainData,"entropy",10,10)
     println("训练完成，所需时间:"+time+"毫秒")
     val AUC=evaluateModel(model,validationData)
     println("评估结果AUC="+AUC)
     return (model)
   }
   def trainModel(trainData:RDD[LabeledPoint],impurity:String,maxDepth:Int,maxBins:Int):(DecisionTreeModel,Double)={
     val startTime=new DateTime()
     val model=DecisionTree.trainClassifier(trainData,2,Map[Int,Int](),impurity,maxDepth,maxBins)
     val endTime=new DateTime()
     val duration=new Duration(startTime,endTime)
     return (model,duration.getMillis())
   }
   def evaluateModel(model:DecisionTreeModel,validationData:RDD[LabeledPoint]):(Double)={
     val scoreAndLabels=validationData.map{data=>
       var predict=model.predict(data.features)
       (predict,data.label)
     }
     val Metrics=new BinaryClassificationMetrics(scoreAndLabels)
     val AUC=Metrics.areaUnderROC
     (AUC)
   }
   def PredictData(sc:SparkContext,model:DecisionTreeModel,categoriesMap:Map[String,Int]):Unit={
     //-------1、导入并转换数据-----------
     val rawDataWithHeader=sc.textFile("data/test.tsv")
     val rawData=rawDataWithHeader.mapPartitionsWithIndex{(idx,iter)=>if(idx==0)iter.drop(1)else iter}
     val lines=rawData.map(_.split("\t"))
     println("共计:"+lines.count.toString()+"条")
     //--------------2.创建训练评估所需的数据RDD[LabeledPoint]---------
    val dataRDD=lines.take(20).map{fields=>
      val trFields=fields.map(_.replaceAll("\"",""))
      val categoryFeaturesArray=Array.ofDim[Double](categoriesMap.size)
      val categoryIdx=categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx)=1
      val numericalFeatures=trFields.slice(4,fields.size).map(d=>if(d=="?")0.0 else d.toDouble)
      val label=0
      //--------------3进行预测-----------
      val url=trFields(0)
      val Features=Vectors.dense(categoryFeaturesArray++numericalFeatures)
      val predict=model.predict(Features).toInt
      var predictDesc={predict match{case 0=>"暂时性网页(ephemeral)";
      case 1=>"长青网页(evergreen)";}}
      println("网址:   "+url+"==>预测:"+predictDesc)  
    }
   }
   def parametersTunning(trainData:RDD[LabeledPoint],validationData:RDD[LabeledPoint]):DecisionTreeModel={
     println("----评估Impurity参数使用gini，entropy-----")
     evaluateParameter(trainData,validationData,"impurity",Array("gini","entropy"),Array(10),Array(10))
     println("-----评估MaxDepth参数使用(3,5,10,15,20)-----")
     evaluateParameter(trainData,validationData,"maxDepth",Array("gini"),Array(3,5,10,15,20,25),Array(10))
     println("-----评估maxBins参数使用(3,5,10,50,100,200)----")
     evaluateParameter(trainData,validationData,"maxBins",Array("gini"),Array(10),Array(3,5,10,50,100,200))
     println("-----所有参数交叉评估做出最好的参数组合-----")
     val bestModel=evaluateAllParameter(trainData,validationData,Array("gini","entropy"),Array(3,5,10,15,20),Array(3,5,10,50,100))
     return (bestModel)
   }
   def evaluateAllParameter(trainData:RDD[LabeledPoint],validationData:RDD[LabeledPoint],impurityArray:Array[String],maxdepthArray:Array[Int],maxBinsArray:Array[Int]):DecisionTreeModel={
     val evaluationsArray=
       for (impurity<-impurityArray;maxDepth<-maxdepthArray;maxBins<-maxBinsArray)
         yield{
         val (model,time)=trainModel(trainData,impurity,maxDepth,maxBins)
         val auc=evaluateModel(model,validationData)
         (impurity,maxDepth,maxBins,auc)
       }
     val BestEval=(evaluationsArray.sortBy(_._4).reverse)(0)
     println("调教后最佳参数:impurity:"+BestEval._1+",maxDepth:"+BestEval._2+",maxBins:"+BestEval._3+",结果AUC="+BestEval._4)
     val (bestModel,time)=trainModel(trainData.union(validationData),BestEval._1,BestEval._2,BestEval._3)
     return bestModel
   }
   def evaluateParameter(trainData:RDD[LabeledPoint],validationData:RDD[LabeledPoint],evaluateParameter:String,impurityArray:Array[String],
       maxdepthArray:Array[Int],maxBinsArray:Array[Int])={
     var dataBarChart=new DefaultCategoryDataset()
     var dataLineChart=new DefaultCategoryDataset()
     for(impurity<-impurityArray;maxDepth<-maxdepthArray;maxBins<-maxBinsArray){
       val(model,time)=trainModel(trainData,impurity,maxDepth,maxBins)
       val auc=evaluateModel(model,validationData)
       val parameterData=evaluateParameter match{
         case "impurity"=>impurity;
         case "maxDepth"=>maxDepth;
         case "maxBins"=>maxBins
       }
       dataBarChart.addValue(auc,evaluateParameter,parameterData.toString())
       dataLineChart.addValue(time,"Time",parameterData.toString())
     }
     Chart.plotBarLineChart("DecisionTree evaluations "+evaluateParameter,evaluateParameter,"AUC",0.58,0.7,"Time",dataBarChart,dataLineChart)
   }
}