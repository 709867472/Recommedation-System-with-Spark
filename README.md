1. Data Source<br>
First we get business and review data from Yelp (https://www.yelp.com/dataset), then we do some filtering and get the input
file input_data.csv and also the 2 test data files for our tasks.

2. Ideas<br>
There are 2 tasks here. The first one is finding similar business pairs according their Jaccard Similarities. The second one
predicting the rating when we input user and business. In task 1, we used MinHashing and Locality Sensitive Hashing. In task2,
we used Spark MLlib, user-based CF and item-based CF.

2. How to run<br>
-python<br>
For task 1: spark-submit find_similar_business.py <path of input_data.csv> <path of output file>
For task 2: spark-submit predict.py <path of input_data.csv> <path of test_data_predictions.csv> <caseId> <path of output file>, where caseId could be integer from 1 to 4. 1 is for MLlib, 2 is for user-based CF, 3 is for item-based CF and 4 is for item-based CF using MinHashing and LSH.<br>
-scala<br>
First we need go to the directory of of myRecommendation.jar: /scala/out/artifacts/myRecommendation, then we type command in terminal.
For task 1: spark-submit --class  find_similar_business.py <path of input_data.csv> <path of output file>

3. Evaluation<br>
For task1, we use evaluation_similar_business.py to output precision and recall, the input for this is the output of task1 and 
test_data_similarities.csv file.
For task2, we use test_data_predictions.csv to output RMSE, the input for this is test_data_predictions.csv and the output of 
task2.
