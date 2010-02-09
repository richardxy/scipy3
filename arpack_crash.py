import numpy as np

import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigen

delnodes =[0, 5, 33, 14, 2, 8 , 186, 10, 25, 16, 57, 51, 95, 92, 30, 53, 
        77, 63, 1 ]
print "reading file"
G = nx.read_adjlist("a_adjlist.txt", create_using=nx.DiGraph(), nodetype=np.int64)# get the directed graph
A = nx.to_scipy_sparse_matrix(G)#now the A is lil sparse matrix
#C = A.tocsc()
#C = C[:, :10000].tocsr()
#R = C[:10000]
R = A.tocsr()#to CSR matrix
for i in [5]:
    print i
    for u, v in G.in_edges_iter(i):
        print u, v
        R[u, v] = 0.0
    for u, v in G.out_edges_iter(i):
        print u, v
        R[u, v] = 0.0
    val = eigen(R, k=6, return_eigenvectors=False)
    vals = filter(lambda x: x.imag == 0.0, val.tolist())
    if vals:
        #filter complex number to get the largest eigenvalue
        print max([x.real for x in vals])
    else:
        print "No real entries", val
