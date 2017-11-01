import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist

from svd_approx import grad_svd
import sklearn.decomposition
from online_lda import LatentDirichletAllocation
import sys
import csv
import random

def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
	reader = csv.reader(open(src_filename), delimiter=delimiter, quoting=quoting)
	colnames = None
	if header:
		colnames = next(reader)
		colnames = colnames[1: ]
	mat = []
	rownames = []
	for line in reader:
		rownames.append(line[0])
		mat.append(np.array(list(map(float, line[1: ]))))
	return (np.array(mat), rownames, colnames)

def build_glove(src_filename):
	return build(src_filename, delimiter=' ', header=False, quoting=csv.QUOTE_NONE)

def glove2dict(src_filename):
	reader = csv.reader(open(src_filename), delimiter=' ', quoting=csv.QUOTE_NONE)
	res =  {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}
	glv_dim = res['the'].shape[0]
	res['<UNK>'] = np.array([random.uniform(-0.5, 0.5) for i in range(glv_dim)])
	return res


# get dataset
data = fetch_20newsgroups(subset='all',remove=['headers', 'footers','quotes'])
vectorizer = CountVectorizer(dtype=np.int32,stop_words='english',max_df=0.8,min_df=1e-3,strip_accents='ascii') #TfidfVectorizer #CountVectorizer
data_t = vectorizer.fit_transform(data.data).tocsr()
X = data_t.todense()
word_to_idx =  vectorizer.vocabulary_
idx_to_word = {v: k for k, v in word_to_idx.items()}

getVec = True
alpha = 2.0
if getVec:
	wordvecs = glove2dict('data/glove.6B.50d.txt')
	vecs = np.zeros((data_t.shape[1],50))
	for k,v in word_to_idx.items():
		q = k if k in wordvecs else '<UNK>'
		vecs[v] = wordvecs[q]
	# word by word size
	dists = cdist(vecs,vecs)
	probs = np.exp(-alpha*dists)
	probs = (probs/probs.sum(0)).T

# fit topic model
models = ['wLDA2']

for model in models:
	for k in [5]:
		if model == 'LDA':
			lda = LatentDirichletAllocation(k,learning_method='online')
			lda.fit(X)
			res = lda.components_
			doc = lda.transform(X)
		elif model == 'wLDA':
			lda = LatentDirichletAllocation(k,learning_method='online')#,word_probs=probs)
			lda.fit(X.dot(probs))
			res = lda.components_
			doc = lda.transform(X)
		elif model == 'wLDA2':
			lda = LatentDirichletAllocation(k,learning_method='online')#,word_probs=probs.T)
			lda.fit(X.dot(probs.T))
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
		for metric in ['cosine','euclidean']:
			confusion_matrix = np.zeros((n_classes,n_classes))
			for idx, row in enumerate(doc):
				#dists = np.sum((row - classes)**2,1)
				dists = pairwise_distances(row.reshape(1,-1),classes,metric=metric)
				c = data.target[idx]
				cp = np.argmin(dists)
				confusion_matrix[c][cp] += 1

			correct = np.diagonal(confusion_matrix).sum()
			total = np.sum(confusion_matrix)
			print("{0} {1} {2} Accuracy {3:.3f}".format(metric,model,k,correct/total))
#plt.imshow(confusion_matrix,interpolation='nearest')
#plt.axis('off')
#plt.show()