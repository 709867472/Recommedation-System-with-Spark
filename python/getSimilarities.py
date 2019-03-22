from pyspark import SparkContext
import random

def mergeDict(dict1, dict2):
    dict1.update(dict2)
    return dict1

def getHashes(partition, userIdMap, numOfUser, hashFunctions):
    hashes = []
    pairs = list(partition)
    for pair in pairs:
        userNum = userIdMap[pair[0]]
        map = {}
        for hashNum in range(1, 61):
            hashFunction = hashFunctions[hashNum]
            a = hashFunction[0]
            b = hashFunction[1]
            map.update({hashNum: (a * userNum + b) % numOfUser})
        hashes.append((pair[0], map))
    return hashes

def getMinHashResult(partition, hashes):
    pairs = list(partition)
    listForMinHash = []
    map = {}
    for pair in pairs:
        userId = pair[0]
        basket = pair[1]
        for businessId in basket:
            for hashNum in range(1, 61):
                key = businessId + ',' + str(hashNum)
                bucketNum =  hashes[userId][hashNum]
                if key in map:
                    if map[key] > bucketNum:
                        map[key] = bucketNum
                else:
                    map[key] = bucketNum
    for key in map:
        listForMinHash.append((key, map[key]))
    return listForMinHash

def changeFormat(partition):
    pairs = list(partition)
    res = []
    map = {}
    for pair in pairs:
        keyPair = pair[0].split(',')
        value = pair[1]
        businessId = keyPair[0]
        hashNum = int(keyPair[1])
        if hashNum not in map:
            map.update({hashNum: {businessId: value}})
        else:
            map[hashNum].update({businessId: value})
    for key in map:
        res.append((key, map[key]))
    return res

def getCandidates(partition, numOfUser, allBusiness):
    minHashes = list(partition)
    res = []
    businessMap = {}
    bucketMap = {}
    for minHash in minHashes:
        basketMap = minHash[1]
        for businessId in allBusiness:
            bit = -1 if businessId not in basketMap else int(basketMap[businessId]) % 1000
            if businessId in businessMap:
                businessMap[businessId] += (',' + str(bit))
            else:
                businessMap[businessId] = str(bit)
    for businessId in businessMap:
        bucketKey = businessMap[businessId]
        if bucketKey not in bucketMap:
            bucketMap.update({bucketKey: []})
        bucketMap[bucketKey].append(businessId)
    for bucketKey in bucketMap:
        if len(bucketMap[bucketKey]) > 1:
            candidates = bucketMap[bucketKey]
            numOfCanidates = len(candidates)
            for i in range(numOfCanidates - 1):
                for j in range(i + 1, numOfCanidates):
                    if candidates[i] < candidates[j]:
                        res.append(candidates[i] + ',' + candidates[j])
                    else:
                        res.append(candidates[j] + ',' + candidates[i])
    return res

def getSimilarities(partition, businessBaskets):
    res = []
    pairs = list(partition)
    for p in pairs:
        pair = p.split(',')
        userSet1 = businessBaskets[pair[0]]
        userSet2 = businessBaskets[pair[1]]
        intersectionNum = len(userSet1.intersection(userSet2))
        unionNum = len(userSet1.union(userSet2))
        similarity = float(intersectionNum) / unionNum
        if similarity >= 0.5:
            res.append((pair[0] + ',' + pair[1], similarity))
    return res

def getFrom(inputPath, onlyCandidates):
    # get textRDD
    sc = SparkContext()
    textRDD = sc.textFile(inputPath)
    header = textRDD.first()

    contentRDD = textRDD.filter(lambda line : line != header)\
                        .map(lambda line: line.split(','))

    businessRDD = contentRDD.map(lambda pair: pair[1])\
                      .distinct()

    userBasketsRDD = contentRDD.map(lambda pair: (pair[0], {pair[1]}))\
                           .reduceByKey(lambda a, b: a | b)

    businessBasketsRDD = contentRDD.map(lambda pair: (pair[1], {pair[0]}))\
                           .reduceByKey(lambda a, b: a | b)

    numOfUser = contentRDD.map(lambda pair: pair[0])\
                          .distinct()\
                          .count()
    allBusiness = businessRDD.collect()

    userIdMap = {}
    i = 1
    for pair in userBasketsRDD.collect():
        userIdMap.update({pair[0] : i})
        i += 1

    hashFunctions = {}
    for hashNum in range(1, 61):
        a = random.randint(1, numOfUser - 1)
        if a % 2 == 0:
            a -= 1
        b = random.randint(0, numOfUser - 1)
        hashFunctions.update({hashNum: (a, b)})

    hashes = userBasketsRDD.mapPartitions(lambda partition: getHashes(partition, userIdMap, numOfUser, hashFunctions))\
                       .collectAsMap()

    minHashResultRDD = userBasketsRDD.mapPartitions(lambda partition: getMinHashResult(partition, hashes))\
                          .reduceByKey(lambda a, b: min(a, b))\
                          .mapPartitions(changeFormat)\
                          .reduceByKey(lambda a, b: mergeDict(a, b))

    candidatesRDD = minHashResultRDD.partitionBy(30, lambda x: int(x % 30))\
                                    .mapPartitions(lambda partition: getCandidates(partition, numOfUser, allBusiness))\
                                    .distinct()
    if onlyCandidates:
        return candidatesRDD.map(lambda candidate: candidate.split(',')).collect()
    businessBaskets = businessBasketsRDD.collectAsMap()
    similaritiesRDD = candidatesRDD.mapPartitions(lambda partition: getSimilarities(partition, businessBaskets))\
                                   .sortByKey()

    similarities = similaritiesRDD.collect()
    sc.stop()
    return similarities
