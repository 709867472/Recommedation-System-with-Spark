from pyspark import SparkContext
import sys

testPath = sys.argv[1]
predictionPath = sys.argv[2]

sc = SparkContext('local[*]', 'evaluate2')
testRDD = sc.textFile(testPath)
predictionRDD = sc.textFile(predictionPath)
header1 = testRDD.first()
header2 = predictionRDD.first()

def getMap(partition):
    res = []
    lines = list(partition)
    for line in lines:
        pair = line.split(',')
        res.append((pair[0] + ',' + pair[1], float(pair[2])))
    return res


testMap = testRDD.filter(lambda line: line != header1)\
                        .mapPartitions(lambda partition: getMap(partition))\
                        .collectAsMap()

predictionMap = predictionRDD.filter(lambda line: line != header2)\
                        .mapPartitions(lambda partition: getMap(partition))\
                        .collectAsMap()

squaresSum = 0
for key in testMap:
    squaresSum += (predictionMap[key] - testMap[key]) ** 2

print('RMSE = ' + str((squaresSum / len(testMap)) ** (1.0 / 2)))
