#INPUTS:
#	B - the M X D codebook
#	X - the N X D matrix whose N rows are each D-dimensional 
#		feature descriptor vectors extracted from an image
#	k - number of nearest neighbors, used in K-nearest neighbors
#	beta - regularization

#both B and X are of type np.matrix, which inherits from np.array, which inherits from list

#OUTPUTS:
#	coeff - the matrix with the image incodings
#  -0.161975   0.222586   0.396576  -0.091430   0.634243
 #  0.681974   0.717732  -0.450878   0.363200  -0.312028
  #-0.549071   0.462220   0.443267  -0.286491   0.930075
   #0.894599   0.659479  -0.516266   0.197584  -0.235396
#last line is bugged

#test data	
#B  = np.matrix([[1 , 2, 3] ,[6, 8, 6], [9, 3, 5], [10, 6, 3], [5, 3, 7] ])
#X = np.matrix([[7, 4, 7],[3, 7, 3],[8, 5, 9],[1, 6, 3]])
#k = 5
#beta = 1e-4
import numpy as np
import operator

def LLC_coding_appr(B, X, k):
	
	beta = 1e-4;
	
	(nframeR, nframeC) = X.shape;
	(nbaseR, nbaseC) = B.shape;
	
	XX = np.multiply(X, X).sum(axis = 1);
	BB = np.multiply(B, B).sum(axis = 1);
	
	D = np.tile(XX, [1, nbaseR]) - 2*X*np.matrix.transpose(B) + np.tile(np.matrix.transpose(BB), [nframeR, 1]);
	
	IDX = np.asmatrix(np.zeros((nframeR, k)));
	for i in range(nframeR):	#watch out, python goes 0...n-1, matlab goes 1...n
		d = D[i];				#d = each row of the matrix D
		
		idx = [];
		for j in sorted(enumerate(d.tolist()[0]), key=operator.itemgetter(1)):
			(a, b) = j;
			idx.append(a);
		IDX[i] = np.asmatrix(idx);
	IDX = IDX.astype(int);
	
	II = np.asmatrix(np.eye(k, dtype=float));
	coeff = np.zeros((nframeR, nbaseR));
	for i in range(nframeR):
		idx = IDX[i][:];
		Bmod = np.matrix(np.zeros(B.shape));
		for j in range(len(idx.tolist()[0])):
			Bmod[j] = B[idx.tolist()[0][j]][:];
		z = Bmod - np.tile(X[i], [k, 1]);
		C = z*np.matrix.transpose(z);
		C = C + II*beta*np.trace(C);
		w = np.linalg.solve(C, np.ones((k, 1)));
		w = w/sum(w);
		for j in range(len(idx.tolist()[0])):
			coeff[i][j] = np.matrix.transpose(w).tolist()[0][idx.tolist()[0][j]];
		#coeff[i] = np.matrix.transpose(w);
	
	return coeff;
		
