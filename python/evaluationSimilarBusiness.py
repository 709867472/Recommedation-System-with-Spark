from pyspark import SparkContext
import sys

outputPath = sys.argv[1]
correctPath = sys.argv[2]

sc = SparkContext('local[*]', 'evaluate1')
outputRDD = sc.textFile(outputPath)
header1 = outputRDD.first()
correctRDD = sc.textFile(correctPath)
header2 = correctRDD.first()


outputContent = outputRDD.filter(lambda line: line != header1)\
                          .map(lambda line: line.split(','))\
                          .map(lambda pair: pair[0] + ',' + pair[1])\
                          .collect()
trainSet = set(outputContent)

correctContent = correctRDD.filter(lambda line: line != header2)\
                     .map(lambda line: line.split(','))\
                     .map(lambda pair: pair[0] + ',' + pair[1])\
                     .collect()
correctSet = set(correctContent)



TP = len(trainSet.intersection(correctSet))
FP = len(trainSet) - TP
FN = len(correctSet) - TP

print('TP = ' + str(TP))
print('FP = ' + str(FP))
print('FN = ' + str(FN))
print('Precision = ' + str(float(TP) / (TP + FP)))
print('Recall = ' + str(float(TP) / (TP + FN)))
