# export PYTHONHASHSEED=323

# importing some libraries
import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkContext
import sys
import time
# import com.github.fommil.netlib:all:1.1.2

sc = SparkContext("local","simple mf 1")
sqlContext = SQLContext(sc)

lines = sc.textFile("s3n://AKIAJVNN43IYNE37I6XQ:fx5s7Fju9A8Xcb5lubS9m+tBwbc8HrTwUFhGpuNI@movielens-ryan/ml-20m/train.csv")
parts = lines.map(lambda l: l.split(','))
# train_df = parts.toDF(["user", "item","rating"])
start_time = time.time()
# importing the MF libraries
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
# reading the movielens data
headers = parts.first()
df_rdd = parts.filter(lambda x: x!=headers)

ratings= df_rdd.map(lambda l: Rating(int(l[0]),int(l[1]),float(l[2])))

X_train, X_test= ratings.randomSplit([0.8, 0.2])
# Training the model
rank = 10
numIterations = 100
model = ALS.train(X_train, rank, numIterations)

# Evaluate the model on testdata
# dropping the ratings on the tests data
testdata = X_test.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
# joining the prediction with the original test dataset
ratesAndPreds = X_test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

# calculating error
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
# print("Mean Squared Error = " + str(MSE))
sys.stdout.write('*****'+"Mean Squared Error = "+str(MSE)+'*******\n')
sys.stdout.flush()
print ('time usage: %s seconds' % (time.time() - start_time))
sys.exit(0)