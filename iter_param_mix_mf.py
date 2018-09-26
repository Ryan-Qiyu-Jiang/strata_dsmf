import collections
import numpy as np
import os
from pyspark import SparkContext
import sys
import time


from pyspark.mllib.linalg.distributed import *

def as_block_matrix(rdd, rowsPerBlock=1024, colsPerBlock=1024):
	return IndexedRowMatrix(
		rdd.zipWithIndex().map(lambda xi: IndexedRow(xi[1], xi[0]))
	).toBlockMatrix(rowsPerBlock, colsPerBlock)

if __name__ == '__main__':

	rank = 20
	num_workers = 3
	num_cycle = 200

	learn_pow = -0.8
	learn_init = 10
	reg_val = 2 

	sc = SparkContext(appName='Distributed Stochastic MF')
	
	#get corpus
	lines_rdd = sc.textFile("s3n://AKIAJVNN43IYNE37I6XQ:fx5s7Fju9A8Xcb5lubS9m+tBwbc8HrTwUFhGpuNI@movielens-ryan/ml-20m/train.csv")
	string_rates_rdd = lines_rdd.map(lambda l: l.split(','))
	headers = string_rates_rdd.first()
	ratings_rdd = string_rates_rdd.filter(lambda x: x!=headers).map(lambda l: (int(l[0]),int(l[1]),float(l[2]),int(l[3])))

	start_time = time.time()

	#get stats, get num users, get num movies, get dict of num for each

	stats = {
		'count':0,
		'num_users':0,
		'num_movies':0,
		'rates_per_user': collections.defaultdict(int),
		'rates_per_movie': collections.defaultdict(int)
	}

	def reduce_counts(state, e):
		u,m,_,_ = e
		state['num_users'] = max(state['num_users'],u)
		state['num_movies'] = max(state['num_movies'],m)
		state['rates_per_user'][u] +=1
		state['rates_per_movie'][m] +=1
		state['count'] +=1
		return state

	def combine_counts(state1, state2):
		state1['num_users'] = max(state1['num_users'],state2['num_users'])
		state1['num_movies'] = max(state1['num_movies'],state2['num_movies'])
		state1['count'] += state2['count']
		for k,v in state2['rates_per_user'].items():
			state1['rates_per_user'][k]+=v

		for k,v in state2['rates_per_movie'].items():
			state1['rates_per_movie'][k]+=v
		return state1

	stats = ratings_rdd.aggregate(stats,reduce_counts,combine_counts)
	num_users = stats['num_users']
	num_movies = stats['num_movies']
	rates_per_user = stats['rates_per_user']
	rates_per_movie = stats['rates_per_movie']
	num_ratings = stats['count']
	print ("RYAN JIANG (stats): ",num_users,":",num_movies,":",num_ratings)

	u_mat = np.random.normal(0.5,0.2,rank*num_users).reshape(num_users,rank).astype(np.float32, copy=False)
	m_mat = np.random.normal(0.5,0.2,rank*num_movies).reshape(num_movies,rank).astype(np.float32, copy=False)

	block_m = int((num_users-1)/num_workers)
	block_n = int((num_movies-1)/num_workers)
		
	def group_blocks(index, some_data):
		
		print("index:",index)
		# permute rows
		def permute(i):
			return (i-index)%num_movies

		block_groups = collections.defaultdict(list)
		for data in some_data:
			block_col = data[0]
			rate_info = data[1]
			u,m,r,uc,mc = rate_info
			block_row = int((u-1)/block_m)
			per_row = permute(block_row)
			block_groups[(per_row,block_row,index)].append(rate_info)

		for block in block_groups.items():
			yield block
	# tag entries with meta data
	ratings_rdd = ratings_rdd.map(lambda x: (int(x[1]/block_m), 
				(x[0],x[1],x[2], 
				stats['rates_per_user'][x[0]], stats['rates_per_movie'][x[1]])))
	
	ratings_rdd = ratings_rdd.partitionBy(num_workers)

	ratings_rdd = ratings_rdd.mapPartitionsWithIndex(group_blocks, preservesPartitioning=True).cache()

	def update(b):
		_, row, col = b[0]
		data = b[1]
		u_block = u_mat[row*block_m:(row+1)*block_m,:]
		m_block = m_mat[col*block_n:(col+1)*block_n,:]

		f=0
		for _ in range(1):
			for u,m,r,uc,mc in data:
				learn_step = pow(learn_init+total+f,learn_pow)
				u_i = int((u-1)%block_m)
				m_i = int((m-1)%block_n)

				d = r - np.dot(u_block[u_i,:],m_block[m_i,:])
				u_grad = -2 * d * m_block[m_i,:]
				u_grad += 2 * reg_val / uc * u_block[u_i,:]
				u_block[u_i,:] -= learn_step * u_grad

				m_grad = -2 * d * u_block[u_i,:]
				m_grad += 2 * reg_val / mc * m_block[m_i,:]
				m_block[m_i,:] -= learn_step * m_grad

				f+=1
		return row,col,u_block,m_block,f

	def loss(user_mat, movies_mat, y):
		squared_error = 0.0
		count = 0
		num_ratings = stats['count']
		h = np.dot(user_mat,movies_mat)
		h_sum = 0
		for k,v in y:
			for u,m,r,_,_ in v:
				d = r - h[u-1,m-1]
				h_sum+=h[u-1,m-1]
				squared_error += d**2
				count+=1
		print ('RYAN JIANG (loss: %f, n: %f, hsum: %f)' % ((squared_error/count),count,h_sum))
		return squared_error/num_ratings
	
	total = 0
	buff = 0
	for i in range(num_cycle):
		if(i%10==0):
			l = loss(u_mat, m_mat.T, ratings_rdd.collect())
			if(l<1):
				print ('time to < 1: %s seconds, cycle %d' % (time.time() - start_time,i))
				print ('time to < 1: %s seconds, cycle %d' % (time.time() - start_time,i))
				print ('time to < 1: %s seconds, cycle %d' % (time.time() - start_time,i))
		u_bmat = sc.broadcast(u_mat)
		m_bmat = sc.broadcast(m_mat)

		print(i, "/",num_cycle)

		grad = ratings_rdd \
			.map(update,preservesPartitioning=True).collect()

		u_bmat.unpersist()
		m_bmat.unpersist()

		for row,col,u_block,m_block,f in grad:
			row_begin = int(row*block_m)
			col_begin = int(col*block_n)

			u_mat[row_begin:row_begin+block_m,:] = u_block
			m_mat[col_begin:col_begin+block_n,:] = m_block
			total+=f

	print ('time usage: %s seconds' % (time.time() - start_time))
	print("RYAN JIANG (factor shapes)",u_mat.shape,m_mat.T.shape)
	print("RYAN JIANG (end loss)",loss(u_mat, m_mat.T, ratings_rdd.collect()))
	sc.stop()

	# 3 nodes, 25 s, 1.3 loss, 10 cycles, 1 sgd step / cycle
	