# wtopic
weighted topic modeling. Experiments in augmenting topic-modeling methods, if given an a priori measurement of the distances between words

# Benchmark
Given the newsgroup dataset, construct a low-dimensional topic embedding. For each newsgroup, construct a mean vector from all documents in it. Evaluation is the percent of documents that are closest (in embedding space) to their correct label. 

# Results
This seems to help both LDA and LSA methods, except for low-dimension LDA embeddings. Notably, the best results on this benchmark came from LSA with tf-idf weights, with the improved method.

| Method 	| k  	| weight 	| distance 	| Improvement 	|
|--------	|----	|--------	|----------	|-------------	|
| LSA    	| 5  	| vec    	| cosine   	| 3.68%       	|
| LSA    	| 10 	| vec    	| cosine   	| 5.17%       	|
| LSA    	| 20 	| vec    	| cosine   	| 10.43%      	|
| LSA    	| 30 	| vec    	| cosine   	| 61.40%      	|
| LDA    	| 5  	| vec    	| euclid   	| -16.73%     	|
| LDA    	| 10 	| vec    	| euclid   	| -5.75%      	|
| LDA    	| 20 	| vec    	| euclid   	| 5.40%       	|
| LDA    	| 30 	| vec    	| euclid   	| 19.55%      	|
| LSA    	| 10 	| idf    	| euclid   	| 6.10%       	|
| LSA    	| 20 	| idf    	| cosine   	| 6.78%       	|
| LSA    	| 30 	| idf    	| cosine   	| 5.35%       	|

# Issues
The method forces parts of these "sparse" algorithms to be dense. This can be augmented by only storing the closest-N pairwise distances for each word. However, this would still make the procedure more dense. 

# Other stuff
There is also a small subset of the newsgroup dataset in here. Along with a script to fetch GLoVe vectors to generating embedding distances. There is also a version of Funk's SVD that I implemented to do approximate LSA, but this didn't really work well
