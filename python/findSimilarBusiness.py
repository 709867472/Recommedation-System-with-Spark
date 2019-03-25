import sys
import time
import getSimilarities

startTime = time.time()

inputPath = sys.argv[1]
output = open(sys.argv[2], 'w')

similarities = getSimilarities.getFrom(inputPath, False)

output.write('business_id_1, business_id_2, similarity\n')
for pair in similarities:
    output.write(pair[0] + ',' + str(pair[1]) + '\n')

endTime = time.time()
print('Duration: ' + str(endTime - startTime))
