import numpy as np
import numpy.matlib
from LLC_coding_appr import *

def LLC_pooling(feaSet, B, pyramid, knn):

	dSize = B.shape[1];
	nSmp = feaSet['feaArr'][0][0].shape[1];

	img_width = feaSet['width'][0][0][0][0];
	img_height = feaSet['height'][0][0][0][0];

	idxBin = np.matrix(np.zeros((nSmp, 1)));

	b = 1e-4;

	# llc coding
	llc_codes = LLC_coding_appr(np.transpose(np.asmatrix(B)), np.transpose(np.asmatrix(feaSet['feaArr'][0][0])), knn, b);
	llc_codes = np.transpose(llc_codes);



	# spatial levels
	pLevels = len(pyramid);

	# spatial bins on each level
	pBins = np.power(pyramid, 2);

	# total spatial bins
	tBins = sum(pBins);

	beta = np.matrix(np.zeros((dSize, tBins)));
	bId = -1;

	for iter1 in range(pLevels):
		nBins = pBins[iter1];
		#print(nBins);
		wUnit = img_width / pyramid[iter1];
		hUnit = img_height / pyramid[iter1];
		xBin = np.ceil(feaSet['x'][0][0] / wUnit);
		yBin = np.ceil(feaSet['y'][0][0] / hUnit);
		idxBin = (yBin - 1) * pyramid[iter1] + xBin;
		for iter2 in range(nBins):
			bId = bId + 1;
		#	print(bId);
			sidxBin = np.where(idxBin == (iter2 + 1))[0].tolist();
			if len(sidxBin) == 0:
				continue;
			beta[:,bId] = np.amax(llc_codes[:,sidxBin], axis=1);
	#print(bId);
	#print(tBins);
	if bId != (tBins -1):
		raise Exception('Index number Error');

	beta = np.transpose(np.matrix(np.ravel(np.transpose(beta))));
	beta = beta / np.sqrt(sum(np.power(beta, 2)));
	return beta;
	