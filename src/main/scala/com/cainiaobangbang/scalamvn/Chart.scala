package com.cainiaobangbang.scalamvn
import org.jfree.chart._
import org.jfree.data.xy._
import org.jfree.data.category.DefaultCategoryDataset
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.axis._
import java.awt.Color
import org.jfree.chart.renderer.category.LineAndShapeRenderer
import org.jfree.chart.plot.DatasetRenderingOrder
import org.jfree.chart.labels.StandardCategoryToolTipGenerator;
import java.awt.BasicStroke
object Chart {
  def plotBarLIneChart(Title:String,xLabel:String,yBarLabel:String,yBarMin:Double,yBarMax:Double,yLineLabel:String,dataBarChart:DefaultCategoryDataset,dataLineChart:DefaultCategoryDataset):Unit={
    //draw Bar Chart
    val chart=ChartFactory.createBarChart("",xLabel,yBarLabel,dataBarChart,org.jfree.chart.plot.PlotOrientation.VERTICAL,true,true,false)
    //get plot
    val plot=chart.getCategoryPlot()
    plot.setBackgroundPaint(new Color(0xEE , 0xEE,0xEE))
    plot.setDomainAxisLocation(AxisLocation.BOTTOM_OR_LEFT)
    plot.setDataset(1,dataLineChart)
    plot.mapDatasetToRangeAxis(1,1)
    //draw y axis
    val vn=plot.getRangeAxis()
    vn.setRange(yBarMin,yBarMax)
    vn.setAutoTickUnitSelection(true)
    //draw line y axis
    val axis2=new NumberAxis(yLineLabel)
    plot.setRangeAxis(1,axis2)
    val renderer2=new LineAndShapeRenderer()
    renderer2.setToolTipGenerator(new StandardCategoryToolTipGenerator())
    //set draw bar first,then draw line
    plot.setRenderer(1,renderer2)
    plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD)
    //create frame
    val frame=new ChartFrame(Title,chart)
    frame.setSize(500,500)
    frame.pack()
    frame.setVisible(true)
  }
}