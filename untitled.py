import numpy as np
import matplotlib.pyplot as plt

# from https://github.com/moorissa/nmf_nyt/
with open('data/nyt_data.txt') as f:
    documents = f.readlines()
documents = [x.strip().strip('\n').strip("'") for x in documents] 

# contains vocabs with rows as index
with open('data/nyt_vocab.dat') as f:
    vocabs = f.readlines()
vocabs = [x.strip().strip('\n').strip("'") for x in vocabs] 

'''create matrix X'''
numDoc = 8447
numWord = 3012 
X = np.zeros([numWord,numDoc])

for col in range(len(documents)):
    for row in documents[col].split(','):
        X[int(row.split(':')[0])-1,col] = int(row.split(':')[1])
Xorig = X.copy()

tfIDF = True

if tfIDF:
	idf = np.log((1.0+np.sum(X > 0, 1))/+1.0)+1.0
	tf = np.log(X,where=X > 0)+1
	X = np.diag(idf).dot(tf)

u,s,vt = np.linalg.svd(X,full_matrices=False)
fullX = u.dot(np.diag(s)).dot(vt)
err = lambda Xh : np.sum((Xh-X)**2)
print(err(fullX))

if False:
	ks = range(10,1000,25)
	errs = [err(u[:,:k].dot(np.diag(s[:k])).dot(vt[:k,:])) for k in ks]
	plt.plot(ks,errs)
	plt.show()
k = 10
Xk = u[:,:k].dot(np.diag(s[:k])).dot(vt[:k,:])
print(err(Xk))

t = 10
for i in range(k):
	maxidx = np.argsort(u[:,i])[::-1]
	print("{} {}".format(i,np.array(vocabs)[maxidx[:t]]))

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(k,learning_method='online')
lda.fit(Xorig)
res = lda.transform(Xorig)
for i in range(k):
	maxidx = np.argsort(res[:,i])[::-1]
	print("{} {}".format(i,np.array(vocabs)[maxidx[:t]]))