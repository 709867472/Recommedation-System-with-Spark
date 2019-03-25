import java.io.PrintWriter
import org.apache.spark.SparkContext
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import util.control.Breaks._

object rui_chen_task2 {
  def main(args: Array[String]): Unit = {
    val startTime = System.currentTimeMillis()
    val trainFile = args(0)
    val testFile = args(1)
    val caseId = args(2)
    val output = new PrintWriter(args(3))

    var sc = new SparkContext("local[*]", "task2")
    val trainRDD = sc.textFile(trainFile)
    val testRDD = sc.textFile(testFile)
    val header1= trainRDD.first()
    val header2 = testRDD.first()

    val trainContentRDD = trainRDD.filter(line => line != header1)
    .map(line => line.split(","))

    val testContentRDD = testRDD.filter(line => line != header2)
        .map(line => line.split(","))

    val userBaskets = trainContentRDD.map(pair => (pair(0), Map[String, Double](pair(1) -> pair(2).toDouble)))
        .reduceByKey((a, b) => a ++ b)
        .collectAsMap()

    val businessBaskets = trainContentRDD.map(pair => (pair(1), Map[String, Double](pair(0) -> pair(2).toDouble)))
        .reduceByKey((a, b) => a ++ b)
        .collectAsMap()


    val testBasketsRDD = testContentRDD.map(pair => (pair(0), pair(1), pair(2).toDouble))

    var predictions: Array[String] = Array()

    if (caseId == "1") {
      var trainUserStrIntMap: scala.collection.mutable.Map[String, Int] = scala.collection.mutable.Map()
      var trainUserIntStrMap: scala.collection.mutable.Map[Int, String] = scala.collection.mutable.Map()
      var i: Int = 1
      for (userId <- userBaskets.keySet) {
        trainUserStrIntMap += (userId -> i)
        trainUserIntStrMap += (i -> userId)
        i += 1
      }
      var trainBusinessStrIntMap: scala.collection.mutable.Map[String, Int] = scala.collection.mutable.Map()
      var trainBusinessIntStrMap: scala.collection.mutable.Map[Int, String] = scala.collection.mutable.Map()
      i = 1
      for (businessId <- businessBaskets.keySet) {
        trainBusinessStrIntMap += (businessId -> i)
        trainBusinessIntStrMap += (i -> businessId)
        i += 1
      }
      val testUserIds = testBasketsRDD.map(pair => pair._1)
          .collect()

      val testBusinessIds = testBasketsRDD.map(pair => pair._2)
          .collect()

      var testUserStrIntMap: scala.collection.mutable.Map[String, Int] = scala.collection.mutable.Map()
      var testUserIntStrMap: scala.collection.mutable.Map[Int, String] = scala.collection.mutable.Map()
      i = 1000000
      for (userId: String <- testUserIds) {
        var intNum = 0
        if (trainUserStrIntMap.contains(userId)) {
          intNum = trainUserStrIntMap(userId)
        } else {
          intNum = i
          i += 1
        }
        testUserStrIntMap += (userId -> intNum)
        testUserIntStrMap += (intNum -> userId)
      }

      var testBusinessStrIntMap: scala.collection.mutable.Map[String, Int] = scala.collection.mutable.Map()
      var testBusinessIntStrMap: scala.collection.mutable.Map[Int, String] = scala.collection.mutable.Map()
      i = 1000000
      for (businessId: String <- testBusinessIds) {
        var intNum = 0
        if (trainBusinessStrIntMap.contains(businessId)) {
          intNum = trainBusinessStrIntMap(businessId)
        } else {
          intNum = i
          i += 1
        }
        testBusinessStrIntMap += (businessId -> intNum)
        testBusinessIntStrMap += (intNum -> businessId)
      }
      val transformedTrainContentRDD = trainContentRDD.mapPartitions(partition => transformStrToInt(partition, trainUserStrIntMap, trainBusinessStrIntMap))
      val trainDataRDD = transformedTrainContentRDD.map(pair => Rating(pair._1, pair._2, pair._3))
      val transformedTestContentRDD = testContentRDD.mapPartitions(partition => transformStrToInt(partition, testUserStrIntMap, testBusinessStrIntMap))
      val testDataRDD = transformedTestContentRDD.map(pair => Rating(pair._1, pair._2, pair._3))

      val rank: Int = 15
      val numIterations: Int = 5
      val model = ALS.train(trainDataRDD, rank, numIterations, 0.15)
      val testInputRDD = testDataRDD.map(pair => (pair.user, pair.product))
      val predictionsRDD = model.predict(testInputRDD).map(r => ((r.user, r.product), r.rating))

      predictions = predictionsRDD.mapPartitions(partition => transformIntToStr(partition, testUserIntStrMap, testBusinessIntStrMap))
        .collect()

      val ratesAndPreds = testDataRDD.map(r => ((r.user, r.product), r.rating)).join(predictionsRDD)
      val MSE = ratesAndPreds.map(r => (r._2._1 - r._2._2) * (r._2._1 - r._2._2)).mean()
      println("Mean Squared Error = " + MSE)
    } else if (caseId == "2") {
      val predictionsRDD = testBasketsRDD.mapPartitions(partition => userBasedPredictions(partition, userBaskets, businessBaskets))
      predictions = predictionsRDD.collect()
    } else if (caseId == "3") {
      val predictionsRDD = testBasketsRDD.mapPartitions(partition => itemBasedPredictions(partition, userBaskets, businessBaskets, Map[String, Set[String]]()))
      predictions = predictionsRDD.collect()
    } else {
      val testBaskets = testBasketsRDD.collect()
      sc.stop()
      val candidates = new getSimilarities().getFrom(trainFile, true)
      sc = new SparkContext()
      val candidateBaskets = sc.parallelize(candidates)
          .mapPartitions(partition => getCandidateBaskets(partition))
          .collectAsMap()

      val predictionsRDD = sc.parallelize(testBaskets).mapPartitions(partition => itemBasedPredictions(partition, userBaskets, businessBaskets, candidateBaskets))
      predictions = predictionsRDD.collect()
    }


    output.write("user_id, business_id, prediction\n")
    for (prediction <- predictions) {
      output.write(prediction + "\n")
    }
    output.close()
    val endTime = System.currentTimeMillis()
    println("Duration: " + (endTime - startTime) / 1000.0)
  }

  def transformStrToInt(partition: Iterator[Array[String]], userStrIntMap: scala.collection.mutable.Map[String, Int], businessStrIntMap: scala.collection.mutable.Map[String, Int]): Iterator[(Int, Int, Double)] = {
    val res: ListBuffer[(Int, Int, Double)] = ListBuffer()
    val pairs = partition.toList
    for (pair <- pairs) {
      val userId = userStrIntMap(pair(0))
      val businessId = businessStrIntMap(pair(1))
      val stars = pair(2).toDouble
      res.append((userId, businessId, stars))
    }

    return res.toIterator
  }

  def transformIntToStr(partition: Iterator[((Int, Int), Double)], userIntStrMap: scala.collection.mutable.Map[Int, String], businessIntStrMap: scala.collection.mutable.Map[Int, String]): Iterator[String] = {
    val res: ListBuffer[String] = ListBuffer()
    val predictions = partition.toList
    for (prediction <- predictions) {
      val userId = userIntStrMap(prediction._1._1)
      val businessId = businessIntStrMap(prediction._1._2)
      val stars = prediction._2
      res.append(userId + "," + businessId + "," + stars)
    }
    return res.toIterator
  }


  def userBasedPredictions(partition: Iterator[(String, String, Double)], userBaskets: scala.collection.Map[String, Map[String, Double]], businessBaskets: scala.collection.Map[String, Map[String, Double]]): Iterator[String] = {
    val res: ListBuffer[String] = ListBuffer()
    val testBaskets = partition.toList
    for (testBasket <- testBaskets) {
      breakable {
        val userId: String = testBasket._1
        val businessId: String = testBasket._2
        if (userBaskets.contains(userId) && userBaskets(userId).contains(businessId)) {
          res.append(userId + "," + businessId + "," + userBaskets(userId)(businessId))
          break
        }
        if (!userBaskets.contains(userId) || !businessBaskets.contains(businessId)) {
          res.append(userId + "," + businessId + "," + 2.5)
          break
        }

        val curStars = userBaskets(userId).values.toList
        val curUserAvg = curStars.sum / curStars.length
        val originalRatedUsers = businessBaskets(businessId).keys.toList
        val curBusinessIds =  userBaskets(userId).keySet

        val intersectedBusiness: scala.collection.mutable.Map[String, Set[String]] = scala.collection.mutable.Map()
        val ratedUsers: ListBuffer[String] = ListBuffer()
        for (ratedUser <- originalRatedUsers) {
          val ratedBusinessIds = userBaskets(ratedUser).keySet
          val intersection = curBusinessIds.intersect(ratedBusinessIds)
          if (intersection.nonEmpty) {
            intersectedBusiness += (ratedUser -> intersection)
            ratedUsers.append(ratedUser)
          }
        }

        val avgs: scala.collection.mutable.Map[String, (Double, Double)] = scala.collection.mutable.Map()
        val curStarsRecord = userBaskets(userId)
        for (ratedUser <- ratedUsers) {
          val ratedStarsRecord = userBaskets(ratedUser)
          val intersection = intersectedBusiness(ratedUser)
          var sum1 = 0.0
          var sum2 = 0.0
          for (intersectedId <- intersection) {
            sum1 += curStarsRecord(intersectedId)
            sum2 += ratedStarsRecord(intersectedId)
            avgs += (ratedUser -> (sum1 / intersection.size, sum2 / intersection.size))
          }
        }


        val diffs: scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, (Double, Double)]] = scala.collection.mutable.Map()
        for (ratedUser <- ratedUsers) {
          val intersection = intersectedBusiness(ratedUser)
          val ratedStarsRecord = userBaskets(ratedUser)
          val curAvg = avgs(ratedUser)._1
          val ratedAvg = avgs(ratedUser)._2
          val localDiffs: scala.collection.mutable.Map[String, (Double, Double)] = scala.collection.mutable.Map()
          for (intersectedId <- intersection) {
            localDiffs += (intersectedId -> (curStarsRecord(intersectedId) - curAvg, ratedStarsRecord(intersectedId) - ratedAvg))
          }
          diffs += (ratedUser -> localDiffs)
        }

        val weights: scala.collection.mutable.Map[String, Double] = scala.collection.mutable.Map()
        for (ratedUser <- ratedUsers) {
          val diffValues = diffs(ratedUser).values.toList
          val d1 = diffValues.map(pair => pair._1 * pair._2).sum
          val d2 = math.pow(diffValues.map(pair => pair._1 * pair._1).sum, 0.5) + math.pow(diffValues.map(pair => pair._2 * pair._2).sum, 0.5)
          if (d2 == 0) {
            weights += (ratedUser -> 0.5)
          } else {
            weights += (ratedUser -> d1 / d2)
          }
        }

        var weightedStarsSum = 0.0
        var weightSum = 0.0
        for (ratedUser <- ratedUsers) {
          weightedStarsSum += (userBaskets(ratedUser)(businessId) - avgs(ratedUser)._2) * weights(ratedUser)
          weightSum += math.abs(weights(ratedUser))
        }
        if (weightSum == 0) {
          res.append(userId + "," + businessId + "," + 2.5)
        } else {
          var predictedStars: Double = curUserAvg + weightedStarsSum / weightSum
          if (predictedStars > 5) {
            predictedStars = 5.0
          } else if (predictedStars < 0) {
            predictedStars = 0.0
          }
          res.append(userId + "," + businessId + "," + predictedStars)
        }
      }
    }
    return res.toIterator
  }


  def itemBasedPredictions(partition: Iterator[(String, String, Double)], userBaskets: scala.collection.Map[String, Map[String, Double]] , businessBaskets: scala.collection.Map[String, Map[String, Double]], candidateBaskets: scala.collection.Map[String, Set[String]]): Iterator[String] = {
    val res: ListBuffer[String] = ListBuffer()
    val testBaskets = partition.toList
    for (testBasket <- testBaskets) {
      breakable {
        val userId = testBasket._1
        val businessId = testBasket._2
        if (businessBaskets.contains(businessId) && businessBaskets(businessId).contains(userId)) {
          res.append(userId + "," + businessId + "," + businessBaskets(businessId)(userId))
          break
        }
        if (!businessBaskets.contains(businessId) || !userBaskets.contains(userId)) {
          res.append(userId + "," + businessId + "," + 2.5)
          break
        }
        var originalRatedBusinesses: Set[String] = userBaskets(userId).keySet
        if (originalRatedBusinesses.isEmpty) {
          res.append(userId + "," + businessId + "," + 2.5)
          break
        }
        val curUserIds = businessBaskets(businessId).keySet
        var intersectedUsers: scala.collection.mutable.Map[String, Set[String]] = scala.collection.mutable.Map()
        val ratedBusinesses: ListBuffer[String] = ListBuffer()
        for (ratedBusiness <- originalRatedBusinesses) {
          val ratedUserIds = businessBaskets(ratedBusiness).keySet
          val intersection = curUserIds.intersect(ratedUserIds)
          if (intersection.nonEmpty) {
            intersectedUsers += (ratedBusiness -> intersection)
            ratedBusinesses.append(ratedBusiness)
          }
        }

        val avgs: scala.collection.mutable.Map[String, (Double, Double)] = scala.collection.mutable.Map()
        val curStarsRecord = businessBaskets(businessId)
        for (ratedBusiness <- ratedBusinesses) {
          val ratedStarsRecord = businessBaskets(ratedBusiness)
          val intersection = intersectedUsers(ratedBusiness)
          var sum1 = 0.0
          var sum2 = 0.0
          for (intersectedId <- intersection) {
            sum1 += curStarsRecord(intersectedId)
            sum2 += ratedStarsRecord(intersectedId)
          }
          avgs += (ratedBusiness -> (sum1 / intersection.size, sum2 / intersection.size))
        }

        val diffs: scala.collection.mutable.Map[String, scala.collection.mutable.Map[String, (Double, Double)]] = scala.collection.mutable.Map()
        for (ratedBusiness <- ratedBusinesses) {
          val intersection = intersectedUsers(ratedBusiness)
          val ratedStarsRecord = businessBaskets(ratedBusiness)
          val curAvg = avgs(ratedBusiness)._1
          val ratedAvg = avgs(ratedBusiness)._2
          val localDiffs: scala.collection.mutable.Map[String, (Double, Double)] = scala.collection.mutable.Map()
          for (intersectedId <- intersection) {
            localDiffs += (intersectedId -> (curStarsRecord(intersectedId) - curAvg, ratedStarsRecord(intersectedId) - ratedAvg))
          }
          diffs += (ratedBusiness -> localDiffs)
        }

        val weights: scala.collection.mutable.Map[String, Double] = scala.collection.mutable.Map()
        for (ratedBusiness <- ratedBusinesses) {
          val diffValues = diffs(ratedBusiness).values.toList
          val d1 = diffValues.map(pair => pair._1 * pair._2).sum
          val d2 = math.pow(diffValues.map(pair => pair._1 * pair._1).sum, 0.5) + math.pow(diffValues.map(pair => pair._2 * pair._2).sum, 0.5)
          if (d2 == 0) {
            weights += (ratedBusiness -> 0.5)
          } else {
            weights += (ratedBusiness -> (d1 / d2) * math.pow(math.abs(d1 / d2), 1.5))
          }
          var similiarBusinesses: Set[String] = Set()
          if (candidateBaskets.contains(businessId)) {
            similiarBusinesses = candidateBaskets(businessId)
          }
          if (similiarBusinesses.contains(ratedBusiness)) {
            weights.update(ratedBusiness, weights(ratedBusiness) * math.pow(math.abs(weights(ratedBusiness)), 0.5))
          }
        }

        var weightedStarsSum = 0.0
        var weightSum = 0.0
        for (ratedBusiness <- ratedBusinesses) {
          weightedStarsSum += businessBaskets(ratedBusiness)(userId) * weights(ratedBusiness)
          weightSum += weights(ratedBusiness)
        }
        if (weightSum == 0) {
          res.append(userId + "," + businessId + "," + 2.5)
        } else {
          var predictedStars: Double = weightedStarsSum / weightSum
          if (predictedStars > 5) {
            predictedStars = 5.0
          } else if (predictedStars < 0) {
            predictedStars = 0.0
          }
          res.append(userId + "," + businessId + "," + predictedStars)
        }
      }
    }
    return res.toIterator
  }

  def getCandidateBaskets(partition: Iterator[(String, Double)]): Iterator[(String, Set[String])] = {
    val res: ListBuffer[(String, Set[String])] = ListBuffer()
    val pairs = partition.toList
    var candidateBaskets: scala.collection.mutable.Map[String, scala.collection.mutable.Set[String]] = scala.collection.mutable.Map()
    for (pair <- pairs) {
      val keyPair = pair._1.split(",")
      val business1 = keyPair(0)
      val business2 = keyPair(1)
      if (!candidateBaskets.contains(business1)) {
        candidateBaskets += (business1 -> scala.collection.mutable.Set[String]())
      }
      candidateBaskets(business1).add(business2)
      if (!candidateBaskets.contains(business2)) {
        candidateBaskets += (business2 -> scala.collection.mutable.Set[String]())
      }
      candidateBaskets(business1).add(business1)
    }
    for (key <- candidateBaskets.keySet) {
      res.append((key, candidateBaskets(key).toSet))
    }
    return res.toIterator
  }
}
