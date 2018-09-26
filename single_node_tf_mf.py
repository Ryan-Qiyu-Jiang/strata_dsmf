import pandas as pd
import numpy as np
import tensorflow as tf

def res_to_mat(res, num_user, num_movie):
    mat = np.zeros((num_user,num_movie))
    for r in res:
        u = int(r[0]-1)
        m = int(r[1]-1)
        s = r[2]
        mat[u][m] = (s-2.5)/2.5
    return mat

import boto3
import io
# s3 = boto3.resource('s3')
# bucket = s3.Bucket('movielens-ryan')
# obj = s3.Object('movielens-ryan', 'ml-20m/train.csv')

s3 = boto3.client('s3')
obj = s3.get_object(Bucket='movielens-ryan', Key='ml-20m/train.csv')
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

train_data = df.fillna(value = 0).drop(['timestamp'],axis=1)
m = train_data.shape[0]
n = train_data.shape[1]

train_y = train_data['rating'].values.reshape((m,1)).T
train_y = (train_y-2.5)/2.5

train_x = train_data.drop(['rating'],axis=1)

train_x = train_x.T
num_pix = train_x.shape[0]
print(train_x.shape,train_y.shape)
temp = m
m = int(m*4/8)

cv_x = train_x.loc[:,m:]
cv_y = train_y[:,m:]
train_x = train_x.loc[:,:m-1]
train_y = train_y[:,:m]

train_users = train_x.loc['userId',:]-1
train_movies = train_x.loc['movieId',:]-1
train_rating = train_y

num_users = train_x[m-1][0]
num_movies = train_x.loc['movieId',:].max()

print("num user:",num_users, "num movie:",num_movies)

#mat_train = res_to_mat(train_data.loc[:m-1,:].values,num_user,num_movie)

print(m,train_x.shape, train_y.shape, cv_x.shape, cv_y.shape)
print(train_x.loc[:,2000:2002])
print(train_y[:,:10])


from numpy import float32

rank = 10
W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.2, mean=0), name="users")
H = tf.Variable(tf.truncated_normal([rank, num_movies], stddev=0.2, mean=0), name="items")

user_bias = np.random.rand(num_users,1)
movie_bias = np.random.rand(1,num_movies)
W_plus_bias = tf.concat( [W, tf.convert_to_tensor(user_bias, dtype=float32, name="user_bias"), tf.ones((num_users,1), dtype=float32, name="item_bias_ones")],1)

H_plus_bias = tf.concat( [H, tf.ones((1, num_movies), name="user_bias_ones", dtype=float32), tf.convert_to_tensor(movie_bias, dtype=float32, name="item_bias")],0)

result = tf.matmul(W_plus_bias, H_plus_bias)
temp_m = train_users.shape[0]
c = [train_users.values,train_movies.values]
print("shapes",train_users.shape, train_movies.shape)
indices = tf.stack(c, axis=1)
print("indices",indices.shape)
result_values = tf.gather_nd(result, indices, name='predicted_rating')
print("result",result_values.shape)
print("diff",result_values.shape, train_rating.shape)
diff_op = result_values-train_rating 


print(W_plus_bias.shape,H_plus_bias.shape,indices.shape,result_values.shape,diff_op.shape, result.shape, train_rating.shape)

loss = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")/m

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1.0
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.96, staircase=True)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

init = tf.global_variables_initializer()

import time
start_time = time.time()
sess = tf.Session()
sess.run(init)
last = 100
for i in range(100):
    _, c = sess.run([optimizer, loss])

    if i%50==0:
        print(i,':',c)
r = sess.run([result_values])
print(train_users[:5],train_movies[:5],r[:5],train_y[:5])
print ('time usage: %s seconds' % (time.time() - start_time))

def acc(h,y):
    n = len(h)
    correct = 0.0
    for i in range(n):
        if abs(h[i]-y[i]) <=(0.5/2.5):
            correct+=1
    return correct/n

show_n = 30
hyp = r[0][show_n-10:show_n]*2.5+2.5
hyp = [round(x,1) for x in hyp]
print(hyp)
print("train rating",train_rating[0][show_n-10:show_n]*2.5+2.5)
last_loss = c = sess.run([loss])
print("acc, loss",acc(r[0],train_rating[0]),last_loss[0])

# 16.1s 1 machine, loss: 0.8

	# u_mat = np.random.normal(0,0.2,rank*num_users).reshape(num_users,rank)
	# m_mat = np.random.normal(0,0.2,rank*num_movies).reshape(num_movies,rank)