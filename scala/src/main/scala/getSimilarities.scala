import org.apache.spark.SparkContext
import org.apache.spark.Partitioner
import scala.collection.mutable.ListBuffer
import scala.util.Random
import scala.math.min

class getSimilarities extends  Serializable {
  def getFrom(inputPath: String, onlyCandidates: Boolean): Array[(String, Double)] = {
    val sc = new SparkContext()
    val textRDD = sc.textFile(inputPath)
    val header = textRDD.first()
    val contentRDD = textRDD.filter(line => line != header)
                            .map(line => line.split(","))

    val businessRDD = contentRDD.map(pair => pair(1))
        .distinct()

    val userBasketsRDD = contentRDD.map(pair => (pair(0), Set(pair(1))))
        .reduceByKey((a, b) => a ++ b)


    val businessBasketsRDD = contentRDD.map(pair => (pair(1), Set(pair(0))))
        .reduceByKey((a, b) => a ++ b)

    val numOfUser: Int = contentRDD.map(pair => pair(0))
        .distinct()
        .count()
      .toInt

    val allBusiness = businessRDD.collect()

    var userIdMap: scala.collection.mutable.Map[String, Int] = scala.collection.mutable.Map()
    var i: Int = 1
    for (pair <- userBasketsRDD.collect()) {
      userIdMap += (pair._1 -> i)
      i += 1
    }

    val hashFunctions: scala.collection.mutable.Map[Int, (Int, Int)] = scala.collection.mutable.Map()
    for (hashNum <- 1 until 61) {
      var a: Int = Random.nextInt(numOfUser - 2) + 1
      if (a % 2 == 0) {
        a -= 1
      }
      val b: Int = Random.nextInt(numOfUser - 1)
      hashFunctions += (hashNum -> (a, b))
    }

    val hashes = userBasketsRDD.mapPartitions(partition => getHashes(partition, userIdMap, numOfUser, hashFunctions))
        .collectAsMap()

    val minHashResultRDD = userBasketsRDD.mapPartitions(partition => getMinHashResult(partition, hashes))
        .reduceByKey((a, b) => min(a, b))
      .mapPartitions(changeFormat)
        .reduceByKey((a, b) => a ++ b)

    val myPartitioner: Partitioner = new Partitioner {
      override def numPartitions: Int = 30

      override def getPartition(key: Any): Int = {
        return key.asInstanceOf[Int] % 30
      }
    }

    val candidatesRDD = minHashResultRDD.partitionBy(myPartitioner)
      .mapPartitions(partition => getCandidates(partition, numOfUser, allBusiness))
      .distinct()

    if (onlyCandidates) {
      val candidates = candidatesRDD.map(candidate => candidate.split(",")).map(pair => (pair(0) + "," + pair(1), pair(2).toDouble)).collect()
      sc.stop()
      return candidates
    }

    val businessBaskets = businessBasketsRDD.collectAsMap()
    val similaritiesRDD = candidatesRDD.mapPartitions(partition => getSimilarityResult(partition, businessBaskets))
        .sortByKey()

    val similarities = similaritiesRDD.collect()
    sc.stop()
    return similarities
  }

  def getHashes(partition: Iterator[(String, Set[String])], userIdMap: scala.collection.mutable.Map[String, Int], numOfUser: Int, hashFunctions: scala.collection.mutable.Map[Int, (Int, Int)]): Iterator[(String, scala.collection.mutable.Map[Int, Int])] = {
    val hashes: ListBuffer[(String, scala.collection.mutable.Map[Int, Int])] = ListBuffer()
    val pairs = partition.toList
    for (pair <- pairs) {
      val userNum: Int = userIdMap(pair._1)
      val map: scala.collection.mutable.Map[Int, Int] = scala.collection.mutable.Map()
      for (hashNum <- 1 until 61) {
        val hashFunction: (Int, Int) = hashFunctions(hashNum)
        val a: Int = hashFunction._1
        val b: Int = hashFunction._2
        map += (hashNum -> ((a * userNum + b) % numOfUser))
        hashes.append((pair._1, map))
      }
    }
    return hashes.toIterator
  }

  def getMinHashResult(partition: Iterator[(String, Set[String])], hashes: scala.collection.Map[String, scala.collection.mutable.Map[Int, Int]]): Iterator[(String, Int)] = {
    val pairs = partition.toList
    val listForMinHash: ListBuffer[(String, Int)] = ListBuffer()
    val minHashMap: scala.collection.mutable.Map[String, Int] = scala.collection.mutable.Map()
    for (pair <- pairs) {
      val userId: String = pair._1
      val basket: Set[String] = pair._2
      for (businessId <- basket) {
        for (hashNum <- 1 until 61) {
          val key: String = businessId + "," + hashNum
          val bucketNum: Int = hashes(userId)(hashNum)
          if (minHashMap.contains(key)) {
            if (minHashMap(key) > bucketNum) {
              minHashMap.update(key, bucketNum)
            }
          } else {
            minHashMap += (key -> bucketNum)
          }
        }
      }
    }
    for (key <- minHashMap.keySet) {
      listForMinHash.append((key, minHashMap(key)))
    }
    return listForMinHash.toIterator
  }


  def changeFormat(partition: Iterator[(String, Int)]): Iterator[(Int, scala.collection.mutable.Map[String, Int])] = {
    val pairs = partition.toList
    val res: ListBuffer[(Int, scala.collection.mutable.Map[String, Int])] = ListBuffer()
    val minHashMap: scala.collection.mutable.Map[Int, scala.collection.mutable.Map[String, Int]] = scala.collection.mutable.Map()
    for (pair <- pairs) {
      val keyPair = pair._1.split(",")
      val value: Int = pair._2
      val businessId: String = keyPair(0)
      val hashNum: Int = keyPair(1).toInt
      if (!minHashMap.contains(hashNum)) {
        minHashMap += (hashNum -> scala.collection.mutable.Map[String, Int](businessId -> value))
      } else {
        minHashMap(hashNum) += (businessId -> value)
      }
    }
    for (key <- minHashMap.keySet) {
      res.append((key, minHashMap(key)))
    }
    return res.toIterator
  }

  def getCandidates(partition: Iterator[(Int, scala.collection.mutable.Map[String, Int])], numOfUser: Int, allBusiness: Array[String]): Iterator[String] = {
    val minHashes = partition.toList
    val res: ListBuffer[String] = ListBuffer()
    val businessMap: scala.collection.mutable.Map[String, String] = scala.collection.mutable.Map()
    val bucketMap: scala.collection.mutable.Map[String, ListBuffer[String]] = scala.collection.mutable.Map()
    for (minHash <- minHashes) {
      val basketMap = minHash._2
      for (businessId <- allBusiness) {
        var bit: Int = -1
        if (basketMap.contains(businessId)) {
          bit = basketMap(businessId) % 1000
        }
        if (businessMap.contains(businessId)) {
          businessMap(businessId) += ("," + bit)
        } else {
          businessMap(businessId) = bit.toString
        }
      }
    }
    for (businessId <- businessMap.keys) {
      val bucketKey = businessMap(businessId)
      if (!bucketMap.contains(bucketKey)) {
        bucketMap += (bucketKey -> ListBuffer())
      }
      bucketMap(bucketKey).append(businessId)
    }
    for (bucketKey <- bucketMap.keys) {
      if (bucketMap(bucketKey).length > 1) {
        val candidates = bucketMap(bucketKey)
        val numOfCanidates = candidates.length
        for (i <- 0 until (numOfCanidates - 1)) {
          for (j <- (i + 1) until numOfCanidates) {
            if (candidates(i) < candidates(j)) {
              res.append(candidates(i) + "," + candidates(j) + ",0.0")
            } else {
              res.append(candidates(j) + "," + candidates(i) + ",0.0")
            }
          }
        }
      }
    }
    return res.toIterator
  }

  def getSimilarityResult(partition: Iterator[String], businessBaskets: scala.collection.Map[String, Set[String]]): Iterator[(String, Double)] = {
    val res: ListBuffer[(String, Double)] = ListBuffer()
    if (partition.nonEmpty) {
      val pairs = partition.toList
      for (p <- pairs) {
        val pair = p.split(",")
        val userSet1 = businessBaskets(pair(0))
        val userSet2 = businessBaskets(pair(1))
        val intersectionNum = userSet1.intersect(userSet2).size
        val unionNum = userSet1.union(userSet2).size
        val similarity = intersectionNum.toDouble / unionNum
        if (similarity >= 0.5) {
          res.append((pair(0) + "," + pair(1), similarity))
        }
      }
    }
    return res.toIterator
  }
}
