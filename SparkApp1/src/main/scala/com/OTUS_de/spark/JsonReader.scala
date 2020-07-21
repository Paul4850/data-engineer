package com.OTUS_de.spark

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{BasicConfigurator, Logger}
import org.json4s.jackson.JsonMethods.parse


import org.apache.log4j.Logger
import org.apache.log4j.Level


import org.json4s._
import org.json4s.jackson.JsonMethods._


//var rdd1 = sc.makeRDD(Array(1, 2, 3, 4))

class JsonReader

object JsonReader {
  def main(args: Array[String]): Unit = {


    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("io").setLevel(Level.INFO)


    val logger = Logger.getLogger(classOf[com.OTUS_de.spark.JsonReader])
    //logger.info("***")
    //BasicConfigurator.configure()

    var conf = new SparkConf()
    conf.setMaster("local")

    conf.setAppName("JsonReader")

    var sc = new SparkContext(conf)
    val rddFromFile = sc.textFile(args(0))

    //var filteredRdd = rddFromFile.zipWithIndex.filter(_._2 < 400)
    //filteredRdd.collect().foreach{println _}

    rddFromFile.map{row =>
      implicit val formats = DefaultFormats
      parse(row).extract[Wine]}
      .foreach{println(_)}
  }
}
