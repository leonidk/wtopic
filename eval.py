import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import svds

from svd_approx import grad_svd
import sklearn.decomposition
from online_lda import LatentDirichletAllocation

# get dataset
data = fetch_20newsgroups(subset='all',remove=['headers', 'footers','quotes'])
vectorizer = CountVectorizer(dtype=np.int32,stop_words='english',max_df=0.8,min_df=1e-3,strip_accents='ascii') #TfidfVectorizer #CountVectorizer
data_t = vectorizer.fit_transform(data.data).tocsr()
X = data_t
word_to_idx =  vectorizer.vocabulary_
idx_to_word = {v: k for k, v in word_to_idx.items()}

# fit topic model
models = ['LSA','LDA']
#if 'LSA' in models:
#	u,s,vt = np.linalg.svd(X,full_matrices=False)

for model in models:
	for k in [5]:
		if model == 'LDA':
			lda = LatentDirichletAllocation(k,learning_method='online')
			lda.fit(X)
			res = lda.components_
			doc = lda.transform(X)
		elif model == 'PMF':
			U,S,V = grad_svd(X,k,1e-4)
			res = V
			doc = U
		else:
			u,s,vt = svds(X.astype(np.float32),k)
			res = vt
			doc = u
		# print topics with top t words
		t = 10
		for i in range(k):
			maxidx = np.argsort(np.abs(res[i,:]))[::-1]
			words = [idx_to_word[idx] for idx in maxidx[:t]]
			#print("{} {}".format(i,words))

		# create center of mass for each class
		n_classes = len(np.unique(data.target))
		classes = np.zeros((n_classes,k))
		counts = np.zeros(n_classes)
		for idx, row in enumerate(doc):
			c = data.target[idx]
			classes[c] += row
			counts[c] += 1
		for c,cnt in zip(classes,counts):
			 c /= float(cnt)

		confusion_matrix = np.zeros((n_classes,n_classes))
		for idx, row in enumerate(doc):
			#dists = np.sum((row - classes)**2,1)
			dists = pairwise_distances(row.reshape(1,-1),classes,metric='cosine')
			c = data.target[idx]
			cp = np.argmin(dists)
			confusion_matrix[c][cp] += 1

		correct = np.diagonal(confusion_matrix).sum()
		total = np.sum(confusion_matrix)
		print("{0} {1} Accuracy {2:.3f}".format(model,k,correct/total))
#plt.imshow(confusion_matrix,interpolation='nearest')
#plt.axis('off')
#plt.show()