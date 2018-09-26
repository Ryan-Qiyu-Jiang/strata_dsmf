# Strata DSMF

*Distributed Stochastic Matrix Factorization using strata optimization*

Parallelized matrix factorization on pyspark.<br>
Trained over Movielens dataset.

## Motivation:

Matrix factorization doesn't shard well, we can't just train mf on shards and average [w][h] as the loss function isn't convex.

Iterative parameter mixing, running mf stochasticly updating [w][h] after each step is slow.

But some pieces of the matrix can be trained totally independently, calling these pieces strata, these strata can be trained without exchanging parameters. This can significantly low network wait time.

*Based on IBM paper https://dl.acm.org/citation.cfm?id=2020426*
