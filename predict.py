# --------------------------------------------------------------------------
# ---  Systems analysis and decision support methods in Computer Science ---
# --------------------------------------------------------------------------
#  Assignment 4: The Final Assignment
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2019
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np

cropped_size = 28
real_size = 36
no_of_labels = 10

def hamming_distance(X_val, X_train):
	"""
	Function calculates Hamming distances between all objects from X_val and all object from X_train.
	Resulting distances are returned as matrix.
	ADAPTED FROM LAB 2
	:param X_val: set of objects that are going to be compared (size: N1xD)
	:param X_train: set of objects compared against param X_val (size: N2xD)
	:return: Matrix of distances between objects X_val and X_train (size: N1xN2)
	"""
	X_train_trans = np.transpose(X_train).astype(int)
	return X_val.shape[1] - X_val @ X_train_trans - (1 - X_val) @ (1 - X_train_trans)

def sort_train_labels_knn(Dist, label):
	"""
	Function sorts labels of training data accordingly to probabilities stored in matrix Dist.
	In each row there are sorted data labels accordingly to corresponding row of matrix Dist.
	ADAPTED FROM LAB 2
	:param Dist: Distance matrix between objects X_val and X_train (size: N1xN2)
	:param label: vector of labels (size: N2)
	:return: Matrix of sorted data labels (with use of mergesort algorithm; size: N1xN2)
	"""
	return label[Dist.argsort(kind='mergesort')]

def p_y_x_knn(y, k):
	"""
	Function calculates conditional probability for all classes and all objects from test set using KNN classifier
	ADAPTED FROM LAB 2
	:param y: matrix of sorted labels for training set (size: N1xN2)
	:param k: number of nearest neighbours
	:return: Matrix of probabilities (size: N1xno_of_labels)
	"""
	prob_matrix = np.zeros(shape=(len(y), no_of_labels))
	for l in range(0, no_of_labels):
		for i in range(len(y)):
			total = 0
			for j in range(0, k):
				if(y[i][j] == l):
					total+=1
			prob_matrix[i][l] = total/k
	return prob_matrix

def pixels_quantization(image):
	"""
	Function quantizates image - values 0, 1
	:param image: array of pixels - values from 0 to 1 (size: cropped_sizexcropped_size)
	:return: array of pixels with values from set (0, 1) (size: cropped_sizexcropped_size)
	"""
	for i in range(0, cropped_size):
		for j in range(0, cropped_size):
			if(image[i][j]<=0.39):
				image[i][j]=int(0)
			else:
				image[i][j]=int(1)
	return image

def frames2(image):
	"""
	Function cropps the image of size 1xreal_size*real_size to size cropped_sizexcropped_size
	:param image: array of pixels (size: real_size*real_size) (one image)
	:return: cropped image - matrix (size: cropped_sizexcropped_size)
	"""
	imag = image.reshape(real_size, real_size)
	x=0
	y=0
	minimum = 10000
	for row in range(0, 8):
		for column in range(0, 8):
			suma = 0
			p = imag[row][column] #pixel value from starting point
			for LtoR in range(0, cropped_size): #one side of realPicture (from left to right)
				suma = suma + np.abs((0+cropped_size)/2 - LtoR)*imag[row][column+LtoR]
			for LtoD in range(0, cropped_size): #from left to down
				suma = suma + np.abs((0+cropped_size)/2 - LtoD)*imag[row+LtoD][column]
			for LtoRD in range(0, cropped_size): #left to right in bottom
				suma = suma + np.abs((0+cropped_size)/2 - LtoRD)*imag[row+cropped_size-1][column+LtoRD]
			for DtoT in range(0, cropped_size): #from down to top
				suma = suma + np.abs((0+cropped_size)/2 - DtoT)*imag[row+DtoT][column+cropped_size-1]
			if(suma<=minimum):
				minimum = suma
				x, y = row, column
	imag = imag[x:(x+cropped_size), y:(y+cropped_size)]
	return imag

def predict(X_test):
	"""
	Function takes images (X_test) as an argument. They are stored in the matrix X_test (size: NxD).
	Function returns a vector y (length: N), where each element of the vector is a class number {0, ..., 9} associated
	with recognized type of cloth.
	:param x: matrix (size: NxD)
	:return: vector (length: N)
	"""
	best_k = 5

	with open('newtrain_01_55000.pkl', 'rb') as f:
		X_train,Y_train = pkl.load(f)

	#X_test = x

	photos = np.zeros(shape = (X_test.shape[0], cropped_size* cropped_size))
	for i in range(0, X_test.shape[0]):
		image = X_test[i]
		image = frames2(image)
		image = pixels_quantization(image)
		image = np.asarray(image)
		photos[i] = image.reshape(cropped_size*cropped_size)

	X_test = photos

	Dist = hamming_distance(X_test, X_train)

	y_sorted = sort_train_labels_knn(Dist, Y_train)

	p_y_x = p_y_x_knn(y_sorted, best_k)

	answer = get_best_column(p_y_x).astype(int)
	answer = np.asarray(answer)
	answer = np.reshape(answer, (-1,1))
	#print(len(answer))
	return answer


def get_best_column(p_y_x):
	"""
	Function returns indexes of columns which have maximum value from row
	:param p_y_x: array of predictions for different labels (size: X_test.shape[0]xno_of_labels)
	:return: Indexes of columns (labels) with maximum value from row
	"""
	answer = np.zeros(len(p_y_x))
	for i in range(0, len(p_y_x)):
		index = 0
		for col in range(1, no_of_labels):
			if(p_y_x[i][col]>=p_y_x[i][index]):
				index = col
		answer[i] = index
	return answer

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final



#with open('train.pkl', 'rb') as f:
    #images,label = pkl.load(f)

'''def typeImage(photo, labels):
    type = 0#for type in range(0, 10):

    equalphotos = []
    equallabels = []
    for type in range(0, 10):
        typeArr = []
        typeLabelArr = []
        for k in range(0, photo.shape[0]):
            if(labels[k]==type and len(typeArr)<235):
                typeArr.append(photo[k])
                typeLabelArr.append(label[k])
                equalphotos.append(photo[k])
                equallabels.append(label[k])
    return equalphotos, equallabels'''

def predictionNoFrame():
	with open('newtrain_01_55000.pkl', 'rb') as f:
		images,label = pkl.load(f)

	valset=10000
	trainset=45000

	X_train = images[valset:valset+trainset]
	Y_train = label[valset:valset+trainset]

	with open('train.pkl', 'rb') as f:
		iiimages,lllabel = pkl.load(f)
	X_test = iiimages[:valset]

	photos = np.zeros(shape = (X_test.shape[0], cropped_size* cropped_size))
	for i in range(0, X_test.shape[0]):
		image = X_test[i]
		image = frames2(image)
		image = pixels_quantization(image)
		image = np.asarray(image)
		photos[i] = image.reshape(cropped_size*cropped_size)


	X_test = photos
	Y_test = lllabel[:valset]

	Dist = hamming_distance(X_test, X_train)
	print('dist', Dist.shape)

	y_sorted = sort_train_labels_knn(Dist, Y_train)
	print('sorted', y_sorted.shape)

	#best_k = 5 #5 -> 0.855
	for best_k in range(3, 50):
		p_y_x = p_y_x_knn(y_sorted, best_k)
		print('p_y_x', p_y_x.shape)
		correct = 0
		for i in range(valset):
			max = 0
			index = 0
			for col in range(0, no_of_labels):
				if(p_y_x[i][col]>=max):
					max = p_y_x[i][col]
					index = col
			realLabel = Y_test[i]
			if(index == realLabel):
				correct = correct + 1
		correctness = correct / valset
		print(str(best_k) +  ": " + str(correctness))


#with open('train.pkl', 'rb') as f:
	#images,label = pkl.load(f)
#print(predict(images[:2000]))
#predictionNoFramesNoNoise()
#predictionNoFrame()




'''def reduce_dataSet():
	with open('train.pkl', 'rb') as f:
		images,label = pkl.load(f)
	with open('trening_750.pkl', 'wb') as f:
		pkl.dump([images[:750], label[:750]], f)'''



'''
I used the following materials:
- Lab2
- https://github.com/markjay4k/Fashion-MNIST-with-Keras
- https://www.youtube.com/watch?v=RJudqel8DVA&list=PLIivdWyY5sqJxnwJhe3etaK7utrBiPBQ2&index=11
- https://nbviewer.jupyter.org/gist/yufengg/2b2fd4b81b72f0f9c7b710fa87077145
- https://github.com/anktplwl91/fashion_mnist/blob/master/fashion_xgboost.ipynb

Noise: https://en.wikipedia.org/wiki/Median_filter
and https://users.soe.ucsc.edu/~milanfar/publications/journal/ModernTour.pdf
'''
