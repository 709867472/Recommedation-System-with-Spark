from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys
import time
import getSimilarities

startTime = time.time()

trainFile = sys.argv[1]
testFile = sys.argv[2]
caseId = sys.argv[3]
output = open(sys.argv[4], 'w')

# get textRDD
sc = SparkContext('local[*]', 'task2')
trainRDD = sc.textFile(trainFile)
testRDD = sc.textFile(testFile)
header1 = trainRDD.first()
header2 = testRDD.first()

def mergeDict(dict1, dict2):
    dict1.update(dict2)
    return dict1

def transformStrToInt(partition, userStrIntMap, businessStrIntMap):
    res = []
    pairs = list(partition)
    for pair in pairs:
        userId = int(userStrIntMap[pair[0]])
        businessId = int(businessStrIntMap[pair[1]])
        stars = float(pair[2])
        res.append((userId, businessId, stars))
    return res

def transformIntToStr(partition, userIntStrMap, businessIntStrMap):
    res = []
    predictions = list(partition)
    for prediction in predictions:
        userId = userIntStrMap[prediction[0][0]]
        businessId = businessIntStrMap[prediction[0][1]]
        stars = prediction[1]
        res.append(userId + ',' + businessId + ',' + str(stars))
    return res

def userBasedPredictions(partition, userBaskets, businessBaskets):
    res = []
    testBaskets = list(partition)
    for testBasket in testBaskets:
        userId = testBasket[0]
        businessId = testBasket[1]
        if userId in userBaskets and businessId in userBaskets[userId]:
            res.append(userId + ',' + businessId + ',' + str(userBaskets[userId][businessId]))
            continue
        if userId not in userBaskets or businessId not in businessBaskets:
            res.append(userId + ',' + businessId + ',' + str(2.5))
            continue
        curStars = list(userBaskets[userId].values())
        curUserAvg = sum(curStars) / len(curStars)
        originalRatedUsers = list(businessBaskets[businessId].keys())
        curBusinessIds = set(userBaskets[userId].keys())
        # get intersected businessIds for each rated user
        intersectedBusiness = {}
        ratedUsers = []
        for ratedUser in originalRatedUsers:
            ratedBusinessIds = set(userBaskets[ratedUser].keys())
            intersection = curBusinessIds.intersection(ratedBusinessIds)
            if len(intersection) > 0:
                intersectedBusiness.update({ratedUser: intersection})
                ratedUsers.append(ratedUser)
        # get average stars for each rated user
        avgs = {}
        curStarsRecord = userBaskets[userId]
        for ratedUser in ratedUsers:
            ratedStarsRecord = userBaskets[ratedUser]
            intersection = intersectedBusiness[ratedUser]
            sum1 = 0.0
            sum2 = 0.0
            for intersectedId in intersection:
                sum1 += curStarsRecord[intersectedId]
                sum2 += ratedStarsRecord[intersectedId]
            avgs.update({ratedUser: (sum1 / len(intersection), sum2 / len(intersection))})
        # get diff between actual stars and average stars for each business of each rated user
        diffs = {}
        for ratedUser in ratedUsers:
            intersection = intersectedBusiness[ratedUser]
            ratedStarsRecord = userBaskets[ratedUser]
            curAvg = avgs[ratedUser][0]
            ratedAvg = avgs[ratedUser][1]
            localDiffs = {}
            for intersectedId in intersection:
                localDiffs.update({intersectedId: (curStarsRecord[intersectedId] - curAvg, ratedStarsRecord[intersectedId] - ratedAvg)})
            diffs.update({ratedUser: localDiffs})
        # get weight of each rated user
        weights = {}
        for ratedUser in ratedUsers:
            diffValues = list(diffs[ratedUser].values())
            d1 = sum(map(lambda pair: pair[0] * pair[1], diffValues))
            d2 = sum(map(lambda pair: pair[0] * pair[0], diffValues)) ** (1.0 / 2) + \
                 sum(map(lambda pair: pair[1] * pair[1], diffValues)) ** (1.0 / 2)
            if d2 == 0:
                weights.update({ratedUser: 0.5})
            else:
                weights.update({ratedUser: d1 / d2})
        # get sum of weighted stars and sum of all the weights
        weightedStarsSum = 0
        weightSum = 0
        for ratedUser in ratedUsers:
            weightedStarsSum += (userBaskets[ratedUser][businessId] - avgs[ratedUser][1]) * weights[ratedUser]
            weightSum += abs(weights[ratedUser])
        if weightSum == 0:
            res.append(userId + ',' + businessId + ',' + str(2.5))
        else:
            predictedStars = curUserAvg + weightedStarsSum / weightSum
            if predictedStars > 5:
                predictedStars = 5.0
            elif predictedStars < 0:
                predictedStars = 0.0
            res.append(userId + ',' + businessId + ',' + str(predictedStars))
    return res


def itemBasedPredictions(partition, userBaskets, businessBaskets, useLSH, candidateBaskets):
    res = []
    testBaskets = list(partition)
    for testBasket in testBaskets:
        userId = testBasket[0]
        businessId = testBasket[1]
        if businessId in businessBaskets and userId in businessBaskets[businessId]:
            res.append(userId + ',' + businessId + ',' + str(businessBaskets[businessId][userId]))
            continue
        if businessId not in businessBaskets or userId not in userBaskets:
            res.append(userId + ',' + businessId + ',' + str(2.5))
            continue
        originalRatedBusinesses = set(userBaskets[userId].keys())
        if useLSH:
            candidateBusinesses = set() if businessId not in candidateBaskets else candidateBaskets[businessId]
            originalRatedBusinesses = originalRatedBusinesses.intersection(candidateBusinesses)
        if len(originalRatedBusinesses) == 0:
            res.append(userId + ',' + businessId + ',' + str(2.5))
            continue
        curUserIds = set(businessBaskets[businessId].keys())
        # get intersected userIds for each rated business
        intersectedUsers = {}
        ratedBusinesses = []
        for ratedBusiness in originalRatedBusinesses:
            ratedUserIds = set(businessBaskets[ratedBusiness].keys())
            intersection = curUserIds.intersection(ratedUserIds)
            if len(intersection) > 0:
                intersectedUsers.update({ratedBusiness: intersection})
                ratedBusinesses.append(ratedBusiness)
        # get average stars for each rated business
        avgs = {}
        curStarsRecord = businessBaskets[businessId]
        for ratedBusiness in ratedBusinesses:
            ratedStarsRecord = businessBaskets[ratedBusiness]
            intersection = intersectedUsers[ratedBusiness]
            sum1 = 0.0
            sum2 = 0.0
            for intersectedId in intersection:
                sum1 += curStarsRecord[intersectedId]
                sum2 += ratedStarsRecord[intersectedId]
            avgs.update({ratedBusiness: (sum1 / len(intersection), sum2 / len(intersection))})
        # get diff between actual stars and average stars for each user of each business user
        diffs = {}
        for ratedBusiness in ratedBusinesses:
            intersection = intersectedUsers[ratedBusiness]
            ratedStarsRecord = businessBaskets[ratedBusiness]
            curAvg = avgs[ratedBusiness][0]
            ratedAvg = avgs[ratedBusiness][1]
            localDiffs = {}
            for intersectedId in intersection:
                localDiffs.update({intersectedId: (curStarsRecord[intersectedId] - curAvg, ratedStarsRecord[intersectedId] - ratedAvg)})
            diffs.update({ratedBusiness: localDiffs})
        # get weight of each rated business
        weights = {}
        for ratedBusiness in ratedBusinesses:
            diffValues = list(diffs[ratedBusiness].values())
            d1 = sum(map(lambda pair: pair[0] * pair[1], diffValues))
            d2 = sum(map(lambda pair: pair[0] * pair[0], diffValues)) ** (1.0 / 2) + \
                 sum(map(lambda pair: pair[1] * pair[1], diffValues)) ** (1.0 / 2)
            if d2 == 0:
                weights.update({ratedBusiness: 0.5})
            else:
                weights.update({ratedBusiness: abs(d1 / d2)})
        # get sum of weighted stars and sum of all the weights
        weightedStarsSum = 0
        weightSum = 0
        for ratedBusiness in ratedBusinesses:
            weightedStarsSum += businessBaskets[ratedBusiness][userId] * weights[ratedBusiness]
            weightSum += weights[ratedBusiness]
        if weightSum == 0:
            res.append(userId + ',' + businessId + ',' + str(2.5))
        else:
            predictedStars = weightedStarsSum / weightSum
            if predictedStars > 5:
                predictedStars = 5.0
            elif predictedStars < 0:
                predictedStars = 0.0
            res.append(userId + ',' + businessId + ',' + str(predictedStars))
    return res

def getCandidateBaskets(partition):
    pairs = list(partition)
    res = []
    candidateBaskets = {}
    for pair in pairs:
        business1 = pair[0]
        business2 = pair[1]
        if business1 not in candidateBaskets:
            candidateBaskets.update({business1: set()})
        candidateBaskets[business1].add(business2)
        if business2 not in candidateBaskets:
            candidateBaskets.update({business2: set()})
        candidateBaskets[business2].add(business1)
    for key in candidateBaskets:
        res.append((key, candidateBaskets[key]))
    return res

trainContentRDD = trainRDD.filter(lambda line : line != header1)\
                          .map(lambda line: line.split(','))
testContentRDD = testRDD.filter(lambda line : line != header2)\
                        .map(lambda line: line.split(','))

userBaskets = trainContentRDD.map(lambda pair: (str(pair[0]), {str(pair[1]): float(pair[2])}))\
                .reduceByKey(lambda a, b: mergeDict(a, b))\
                .collectAsMap()

businessBaskets = trainContentRDD.map(lambda pair: (str(pair[1]), {str(pair[0]): float(pair[2])}))\
                .reduceByKey(lambda a, b: mergeDict(a, b))\
                .collectAsMap()

testBasketsRDD = testContentRDD.map(lambda pair: (str(pair[0]), str(pair[1]), float(pair[2])))

predictions = []
if caseId == '1':
    trainUserStrIntMap = {}
    trainUserIntStrMap = {}
    i = 1
    for userId in userBaskets.keys():
        trainUserStrIntMap.update({userId: i})
        trainUserIntStrMap.update({i: userId})
        i += 1
    trainBusinessStrIntMap = {}
    trainBusinessIntStrMap = {}
    i = 1
    for businessId in businessBaskets.keys():
        trainBusinessStrIntMap.update({businessId: i})
        trainBusinessIntStrMap.update({i: businessId})
        i += 1
    testUserIds = testBasketsRDD.map(lambda pair: pair[0])\
                                .collect()
    testBusinessIds = testBasketsRDD.map(lambda pair: pair[1])\
                                .collect()
    testUserStrIntMap = {}
    testUserIntStrMap = {}
    i = 1000000
    for userId in testUserIds:
        intNum = 0
        if userId in trainUserStrIntMap:
            intNum = trainUserStrIntMap[userId]
        else:
            intNum = i
            i += 1
        testUserStrIntMap.update({userId:  intNum})
        testUserIntStrMap.update({intNum:  userId})

    testBusinessStrIntMap = {}
    testBusinessIntStrMap = {}
    i = 1000000
    for businessId in testBusinessIds:
        intNum = 0
        if businessId in trainBusinessStrIntMap:
            intNum = trainBusinessStrIntMap[businessId]
        else:
            intNum = i
            i += 1
        testBusinessStrIntMap.update({businessId:  intNum})
        testBusinessIntStrMap.update({intNum:  businessId})
    transformedTrainContentRDD = trainContentRDD.mapPartitions(lambda partition: transformStrToInt(partition, trainUserStrIntMap, trainBusinessStrIntMap))
    trainDataRDD = transformedTrainContentRDD.map(lambda pair: Rating(pair[0], pair[1], float(pair[2])))
    transformedTestContentRDD = testContentRDD.mapPartitions(lambda partition: transformStrToInt(partition, testUserStrIntMap, testBusinessStrIntMap))
    testDataRDD = transformedTestContentRDD.map(lambda pair: Rating(pair[0], pair[1], float(pair[2])))
    # Build the recommendation model using Alternating Least Squares
    rank = 15
    numIterations = 5
    model = ALS.train(trainDataRDD, rank, numIterations, 0.15)

    testInputRDD = testDataRDD.map(lambda pair: (pair[0], pair[1]))
    predictionsRDD = model.predictAll(testInputRDD).map(lambda r: ((r[0], r[1]), r[2]))
    predictions = predictionsRDD.mapPartitions(lambda partition: transformIntToStr(partition, testUserIntStrMap, testBusinessIntStrMap))\
                                .collect()

    ratesAndPreds = testDataRDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictionsRDD)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))

elif caseId == '2':
    predictionsRDD = testBasketsRDD.mapPartitions(lambda partition: userBasedPredictions(partition, userBaskets, businessBaskets))
    predictions = predictionsRDD.collect()
elif caseId == '3':
    predictionsRDD = testBasketsRDD.mapPartitions(lambda partition: itemBasedPredictions(partition, userBaskets, businessBaskets, False, {}))
    predictions = predictionsRDD.collect()
else:
    testBaskets = testBasketsRDD.collect()
    sc.stop()
    candidates = getSimilarities.getFrom(trainFile, True)
    sc = SparkContext()
    candidateBaskets =  sc.parallelize(candidates)\
                         .mapPartitions(lambda partition: getCandidateBaskets(partition))\
                         .collectAsMap()
    predictionsRDD = sc.parallelize(testBaskets).mapPartitions(lambda partition: itemBasedPredictions(partition, userBaskets, businessBaskets, True, candidateBaskets))
    predictions = predictionsRDD.collect()

output.write('user_id, business_id, stars\n')
for prediction in predictions:
    output.write(prediction + '\n')

endTime = time.time()
print('Duration: ' + str(endTime - startTime))
