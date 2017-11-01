import numpy as np

def grad_svd(A,rank,lr):
	d, t = A.shape
	P = np.random.randn(d,rank)
	Q = np.random.randn(rank,t)
	eo = np.ones((1,1))*1e12
	while True:
		e = A - P.dot(Q)
		P += lr * e.dot(Q.T)
		Q += lr * P.T.dot(e)
		diff = np.linalg.norm(eo) - np.linalg.norm(e)
		if diff < lr:
			break
		eo = e
		#print(diff)
	S1 =np.sqrt(np.sum(P**2,0))
	S2 =np.sqrt(np.sum(Q**2,1)) 

	U = P/S1
	V = (Q.T/S2).T
	S = S1 * S2
	return U,S,V

if __name__ == '__main__':
	A = np.random.randn(10,20)
	u,s,vt = np.linalg.svd(A,full_matrices=False)
	err = lambda X,Xh: np.sum((Xh-X)**2)
	k = 5
	print("full error",err(A,u.dot(np.diag(s).dot(vt))))
	print("low-rank error",err(A,u[:,:k].dot(np.diag(s[:k])).dot(vt[:k,:])))
	full_err = err(A,u[:,:k].dot(np.diag(s[:k])).dot(vt[:k,:]))
	
	U,S,V = grad_svd(A,5,1e-3)

	print("low-rank-approx error",err(A,U.dot(np.diag(S).dot(V))))