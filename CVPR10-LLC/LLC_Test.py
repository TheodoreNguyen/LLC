import numpy as np
import scipy.io
import os
from LLC_pooling import *
from sklearn import svm
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning);

#set initial vars
pyramid = [1, 2, 4];
knn = 5;
c = 10;
nRounds = 10;
tr_num = 30;
mem_block = 3000;

#set strings for the directory the images/descriptors/features will be in
img_dir = 'image';
data_dir = 'data';
fea_dir = 'features';
test_dir = 'test';		#directory the SIFT descriptors from images to be tested

codebook_dir = 'dictionary';
codebookName = 'Caltech101_SIFT_Kmeans_1024.mat';

#SIFT database should already be in data_dir. If not, run extr_sift(img_dir, data_dir)
#retrieving the sift database: database = retr_database_dir(data_dir);

#create methods to index the data - classes is an array whose elements are just
#strings for each class. label is a dictionary whose keys are the classes, and 
#whose elements of the key are the names of images in that class. path is the
#same as label, except instead of the names of images, we have the path of the 
#image.
classes = []; 
label = {};
path = {};
fpath = {};
fea = [];
feaclass = [];
numImages = 0;

subfolders = os.listdir(data_dir);

for folder in subfolders:
	if ((folder != '.') & (folder != '..')):
		classes.append(folder);

for category in classes:
	label[category] = [];
	path[category] = [];
	fpath[category] = [];
	for file in os.listdir(os.path.join(data_dir, category)):
			if file.endswith('.mat'):
					numImages = numImages + 1;
					label[category].append(file);
					path[category].append(os.path.join(data_dir, category, file));
					fpath[category].append(os.path.join(fea_dir, category, file));
#load the codebook 
Braw = scipy.io.loadmat(os.path.join(codebook_dir, codebookName));
B = Braw['B'];
nCodebook = B.shape[1];

print "Finished loading codebook and extracting dirs.\n"

#create the final image representation vectors and save them

for category in classes:
	if not os.path.exists(os.path.join(fea_dir, category)):
		os.makedirs(os.path.join(fea_dir, category));
	for i in range(len(label[category])):
		feaSet = scipy.io.loadmat(path[category][i])['feaSet'];
		fea.append(LLC_pooling(feaSet, B, pyramid, knn));
		feaclass.append(category);
		scipy.io.savemat(os.path.join(fea_dir, category, label[category][i]), mdict = {'fea':fea[i]});
		
print "Finished creating final image rep vectors for the training set, and saving them.\n"

#fit the model
vector = [];
for vec in fea:
	vector.append(np.transpose(vec).tolist()[0]);
	
lin_clf = svm.LinearSVC();
lin_clf.fit(vector, feaclass);

print "Finished training the model.\n"

#--------------------------------------------------------

#load SIFT descriptors for images you want to test

total = 0;
correct = 0;
featest = [];
truecat = [];
if len(os.listdir(test_dir)) == 0:
	print "There are no directories, therefore no SIFT descriptors here.\n";
else:
	for category in os.listdir(test_dir):
		for SIFTs in os.listdir(os.path.join(test_dir, category)):
			if SIFTs.endswith('.mat'):
				total = total + 1;
				feaSet = scipy.io.loadmat(os.path.join(test_dir, category, SIFTs))['feaSet'];
				featest.append(LLC_pooling(feaSet, B, pyramid, knn));
				truecat.append(category);
	print "Finished created final image rep vectors for the testing set.\n"
	vector2 = [];
	for vec in featest:
		vector2.append(np.transpose(vec).tolist()[0]);
	for i in range(len(vector2)):
		if(lin_clf.predict(vector2[i]) == truecat[i]):
			correct = correct + 1;
	
	proportion = float(correct)/float(total) * 100;
	print "Summary statistics of the model:"
	print "%i classes were present in training of the model," % len(classes)
	print "with a total of %i SIFT descriptors in those classes for training." % numImages
	print "There was a total of %i test SIFT descriptors." % total
	print "Of those, %i of them were classified correctly." % correct
	print "This results in an accuracy of %f percent." % proportion
