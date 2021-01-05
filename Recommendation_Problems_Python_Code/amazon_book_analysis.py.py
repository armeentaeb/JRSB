 #mport appropriate libraries
import sys
import shutil
import os.path

import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark import SQLContext, SparkConf, SparkContext
import random
from pyspark.mllib.linalg import Matrix, Matrices
from random import randint
import time
from pyspark.sql import SparkSession, Row
import timeit


def toTextFile(data):
	return '::'.join(str(d) for d in data)



from pyspark.sql import DataFrame
from functools import reduce  # For Python 3.x


def unionAll(*dfs):
	return reduce(DataFrame.unionAll, dfs)


# execute ALS
def ALSSpark(nUser,nItem,ratingsDF,rank, maxIter, lam):

	
	sc.setCheckpointDir('~/checkpoints')

	als = ALS(rank=rank, maxIter=maxIter, regParam=lam)
	model = als.fit(ratingsDF)


	Uest = model.userFactors.orderBy("id").collect()
	Uest = np.array([x[1] for x in Uest])
	Vest = model.itemFactors.orderBy("id").collect()
	Vest = np.array([x[1] for x in Vest])


	return (Uest,Vest,model)	





def selectedUserMostObserved(ratings,userThresh,itemThresh):

	pairUser = ratings.map(lambda l: (l[0],(l[1],float(l[2]))))
	userDict = pairUser.aggregateByKey([], lambda a,b: a + [b],lambda x, y: x+y).collectAsMap()
	userID = list(userDict.keys())
	data = []
	for i in range(0,len(userID)):
		t = userDict[userID[i]]
		if len(t) > userThresh:
			for j in range(0,len(t)):
				data.append((userID[i],t[j][0],t[j][1]))

	ratings = sc.parallelize(data)
	
	pairItem = ratings.map(lambda l: (l[1],(l[0],float(l[2]))))
	itemDict = pairItem.aggregateByKey([], lambda a,b: a + [b],lambda x, y: x+y).collectAsMap()
	itemID = list(itemDict.keys())
	data = []
	for i in range(0,len(itemID)):
		t = itemDict[itemID[i]]
		if len(t) > itemThresh:
			for j in range(0,len(t)):
				data.append((t[j][0],itemID[i],t[j][1]))

	ratings = sc.parallelize(data)
		

	userMap = ratings.\
		map(lambda x: x[0]).\
		distinct().\
		zipWithIndex().\
		collectAsMap()

	itemMap = ratings.\
		map(lambda x: x[1]).\
		distinct().\
		zipWithIndex().\
		collectAsMap()


	ratings = ratings.map(lambda l: (userMap[l[0]],itemMap[l[1]],l[2]))

	return ratings				



def saveFile(objects,fileName):
	#dictS = []
	#for k in range(0,len(objects)):
		#dictS.append((k,np.asscalar(objects[k])))
	#sc.parallelize(dictS).repartition(1).toDF(['index','val']).write.json(fileName)
	f = open(fileName,'w')
	for k in range(0,len(objects)):
		string = str(objects[k])+'\n'
		f.write(string)
	f.close()	

 

def saveFileSing(objects,fileName):

	dictS = []
	t = objects.shape
	L = open(fileName,'w')

	for h in range(0,t[1]):		
		for f in range(0,t[0]):
			#print(objects[f][h])
			string = str(objects[f][h])+'\n'
			L.write(string)
	L.close() 



def biasRemoval(ratings,outfile):

	pairUser = ratings.map(lambda l: (l[0],(l[1],float(l[2]))))
	print(ratings.count())
	userDict = pairUser.aggregateByKey([], lambda a,b: a + [b],lambda x, y: x+y).collectAsMap()
	userID = list(userDict.keys())
	with open('ratingsProcessed.dat', "w") as f:
		for i in range(0,len(userID)):
			t = userDict[userID[i]]
			l1 = [int(x[1]) for x in t]
			avgRating = np.mean(np.array(l1))
			for j in range(0,len(t)):
				st = [userID[i],userDict[userID[i]][j][0],userDict[userID[i]][j][1]- avgRating] 
				str1 = '::'.join(str(e) for e in st)
				f.write( '%s\n' %(str1))
	f.close()



# this function performs subspace stability selection
def Stability(numUsers,numItems,pairTrain,rank,numBags,lam):

	print(numUsers)
	sumColSpace2 = np.zeros((numUsers,numUsers))		# this is the user subspace	
	sumRowSpace2 = np.zeros((numItems,numItems))		# this is the item subspace

	sumColSpace3 = np.zeros((numUsers,numUsers))		# this is the user subspace	
	sumRowSpace3 = np.zeros((numItems,numItems))

	for i in range(0,numBags):
		DataSamples,crap = createTrainData(pairTrain,0.5)

		df = DataSamples.map(lambda l: Row(user = (l[0]), item=(l[1]), rating=l[2])).toDF()
		U1,V1,model = ALSSpark(numUsers,numItems,df,rank, maxIter, lam)

		Unew,s,Vn = np.linalg.svd(U1,full_matrices = True)
		#print(np.linalg.norm((sumColSpace2 + np.dot(Unew[:,0:rank],Unew[:,0:rank].T))/(i+1) -sumColSpace2/(i),'fro'))

		sumColSpace2 = sumColSpace2 + np.dot(Unew[:,0:rank],Unew[:,0:rank].T)

		Unew = np.dot(Unew[:,0:rank],np.diag(np.sqrt(s[0:rank])))
		sumColSpace3 = sumColSpace3 + np.dot(Unew[:,0:rank],Unew[:,0:rank].T)


		Vnew,s,Vn = np.linalg.svd(V1,full_matrices = True)
		sumRowSpace2 = sumRowSpace2 + np.dot(Vnew[:,0:rank],Vnew[:,0:rank].T)
		Vnew = np.dot(Vnew[:,0:rank],np.diag(np.sqrt(s[0:rank])))
		sumRowSpace3 = sumRowSpace3 + np.dot(Vnew,Vnew.T)






	Ureg,sCol,V = np.linalg.svd(sumColSpace2/(numBags),full_matrices = False)
	Vreg,sRow,V = np.linalg.svd(sumRowSpace2/(numBags),full_matrices = False)
	
	Uscaled,s,V = np.linalg.svd(sumColSpace3/(numBags),full_matrices = False)
	Vscaled,s,V = np.linalg.svd(sumRowSpace3/(numBags),full_matrices = False)
	


	return (Ureg,sCol,Vreg,sRow,Uscaled,Vscaled)




# this function computes training validation and testing performance
def computePerformance(pairTrain,pairValid,pairTest,Ues,sCol,Ves,sRow,Uscaled,Vscaled,thresh):

	rankEst1 = sum(i > thresh for i in sCol)
	rankEst2 = sum(i > thresh for i in sRow)
	rankEs = min([rankEst1,rankEst2])
	U0 = Ues[:,0:rankEs]
	V0 = Ves[:,0:rankEs]
	Uscaled = Uscaled[:,0:rankEs]
	Vscaled = Vscaled[:,0:rankEs]
	sizeGen = pairTest.count()
	sizeValid = pairValid.count()
	train_size = pairTrain.count()

	if rankEs == 0:
		testSt = pairTest.map(lambda t: np.square(t[2]-0)).reduce(lambda x,y: x+y)/sizeGen
		validSt = pairValid.map(lambda t: np.square(t[2])).reduce(lambda x,y: x+y)/sizeValid
		print("generalization error Stability for alpha = %f is %f" %(thresh,testSt))
		print("validation error Stability for alpha = %f is %f" %(thresh,validSt))
		print("the rank is %f" %(rankEs))
	else:
		print("the rank is %f" %(rankEs))
		# computing the validation and generalization performance of standard approach
		sum1 = pairTrain.sample(False,1).map(lambda x: np.kron(np.dot(V0[int(x[1]),:].reshape(rankEs,1),V0[int(x[1]),:].reshape(1,rankEs)), np.dot(U0[int(x[0]),:].reshape(rankEs,1),U0[int(x[0]),:].reshape(1,rankEs)))).reduce(lambda x,y: x+y)
		sum2 = pairTrain.sample(False,1).map(lambda x: np.dot(U0[int(x[0]),:].reshape(rankEs,1),V0[int(x[1]),:].reshape(1,rankEs)*x[2])).reduce(lambda x,y: x+y)
		M = np.dot(np.linalg.inv(sum1),sum2.T.reshape(rankEs*rankEs,1))
		M = M.reshape(rankEs,rankEs).T
		testSt = pairTest.map(lambda t: np.square(t[2]-np.dot(np.dot(U0[int(t[0]),:],M),V0[int(t[1]),:].T))).reduce(lambda x,y: x+y)/sizeGen
		validSt = pairValid.map(lambda t: np.square(t[2]-np.dot(np.dot(U0[int(t[0]),:],M),V0[int(t[1]),:].T))).reduce(lambda x,y: x+y)/sizeValid
		train_error = pairTrain.map(lambda t: np.square(t[2]-np.dot(np.dot(U0[int(t[0]),:],M),V0[int(t[1]),:].T))).reduce(lambda x,y: x+y)/train_size

		print("holdout error Stability for alpha = %f is %f" %(thresh,testSt))
		print("validation error Stability for alpha = %f is %f" %(thresh,validSt))
		print("generalization error Stability for alpha = %f is %f" %(thresh,testSt-train_error))
					


		#print("generalization error Stability for alpha = %f is %f" %(thresh,testSt))
		#print("validation error Stability for alpha = %f is %f" %(thresh,validSt))
				

		# computing the validation and generalization performance with scaled
		sum1 = pairTrain.sample(False,1).map(lambda x: np.kron(np.dot(Vscaled[int(x[1]),:].reshape(rankEs,1),Vscaled[int(x[1]),:].reshape(1,rankEs)), np.dot(Uscaled[int(x[0]),:].reshape(rankEs,1),Uscaled[int(x[0]),:].reshape(1,rankEs)))).reduce(lambda x,y: x+y)
		sum2 = pairTrain.sample(False,1).map(lambda x: np.dot(Uscaled[int(x[0]),:].reshape(rankEs,1),Vscaled[int(x[1]),:].reshape(1,rankEs)*x[2])).reduce(lambda x,y: x+y)
		M = np.dot(np.linalg.inv(sum1),sum2.T.reshape(rankEs*rankEs,1))
		M = M.reshape(rankEs,rankEs).T
		testScSt = pairTest.map(lambda t: np.square(t[2]-np.dot(np.dot(Uscaled[int(t[0]),:],M),Vscaled[int(t[1]),:].T))).reduce(lambda x,y: x+y)/sizeGen
		validScSt = pairValid.map(lambda t: np.square(t[2]-np.dot(np.dot(Uscaled[int(t[0]),:],M),Vscaled[int(t[1]),:].T))).reduce(lambda x,y: x+y)/sizeValid
		print("generalization error scaled Stability for alpha = %f is %f" %(thresh,testScSt))
		print("validation error Stability for alpha = %f is %f" %(thresh,validScSt))

	return (testSt,validSt,testSt,validSt,rankEs)




# this function executes the vanilla approach and subspace stability selection
def rankEstimation(pairTrain,pairTest,pairValid,numUsers,numItems,numBags,embedDim):



	size_Gen = pairTest.count()
	size_Valid = pairValid.count()
	size_Train = pairTrain.count()
	lam_vector = np.linspace(0.01,0.6,45)
	#lam_vector = [0.5]

	
	MSE_valid_7 = np.zeros((len(lam_vector),1))
	MSE_test_7 = np.zeros((len(lam_vector),1))
	MSE_valid_8 = np.zeros((len(lam_vector),1))
	MSE_test_8 = np.zeros((len(lam_vector),1))
	MSE_valid_9 = np.zeros((len(lam_vector),1))
	MSE_test_9 = np.zeros((len(lam_vector),1))
	MSE_vanilla_valid = np.zeros((len(lam_vector),1))
	MSE_vanilla_test = np.zeros((len(lam_vector),1))
	rank_stability_7 = np.zeros((len(lam_vector),1))
	rank_stability_8 = np.zeros((len(lam_vector),1))
	rank_stability_9 = np.zeros((len(lam_vector),1))
	rank_vanilla = np.zeros((len(lam_vector),1))

	FileName = '/Users/ataeb/Documents/Amazon_Video_Games/results/'

	column_sing = np.zeros((numUsers,len(lam_vector)))
	row_sing = np.zeros((numItems,len(lam_vector)))
	sing_vanilla = np.zeros((embedDim,len(lam_vector)))
	train_size = pairTrain.count()
	
	
	for i in range(0,len(lam_vector)):

		print("----------------------------------")
		lam = lam_vector[i]
		print("the lambda analyzed is %f" %(lam))
		rank = embedDim
		U0,sCol,V0,sRow,Uscaled,Vscaled = Stability(numUsers,numItems,pairTrain,embedDim,numBags,lam)		
		(MSE_valid_7[i],MSE_test_7[i],crap,crap,rank_stability_7[i]) = computePerformance(pairTrain,pairValid,pairTest,U0,sCol,V0,sRow,Uscaled,Vscaled,0.7)
		(MSE_valid_8[i],MSE_test_8[i],crap,crap,rank_stability_8[i]) = computePerformance(pairTrain,pairValid,pairTest,U0,sCol,V0,sRow,Uscaled,Vscaled,0.8)
		(MSE_valid_9[i],MSE_test_9[i],crap,crap,rank_stability_9[i]) = computePerformance(pairTrain,pairValid,pairTest,U0,sCol,V0,sRow,Uscaled,Vscaled,0.9)
		df = pairTrain.map(lambda l: Row(user = (l[0]), item=(l[1]), rating=l[2])).toDF()
		U1,V1,model = ALSSpark(numUsers,numItems,df,embedDim, 30, lam)
		u1,t1,v1 = np.linalg.svd(np.dot(U1,V1.T),full_matrices = False)
		rank_vanilla[i] = sum(j > 0.001 for j in t1)
		sing_vanilla[0:embedDim,i] = t1[0:embedDim]
		MSE_vanilla_test[i] = pairTest.map(lambda t: np.square(t[2]-np.dot(U1[int(t[0]),:],V1[int(t[1]),:].T))).reduce(lambda x,y: x+y)/size_Gen
		MSE_vanilla_valid[i] = pairValid.map(lambda t: np.square(t[2]-np.dot(U1[int(t[0]),:],V1[int(t[1]),:].T))).reduce(lambda x,y: x+y)/size_Valid
		train_error = pairTrain.map(lambda t: np.square(t[2]-np.dot(U1[int(t[0]),:],V1[int(t[1]),:].T))).reduce(lambda x,y: x+y)/train_size

		column_sing[0:numUsers,i] = sCol
		row_sing[0:numItems,i] = sRow		
		print("the rank of the vanilla approach is %f" %(rank_vanilla[i]))
		print("hold out error vanilla is %f" %(MSE_vanilla_test[i]))
		print("validation error vanilla is %f" %(MSE_vanilla_valid[i]))
		print("generalization error is %f" %(MSE_vanilla_test[i]-train_error))
		saveFileSing(sing_vanilla,FileName+'/sing_vanilla'+'.txt')

		

	saveFileSing(column_sing,FileName+'/column_sing'+'.txt')
	saveFileSing(sing_vanilla,FileName+'/sing_vanilla'+'.txt')
	saveFileSing(row_sing,FileName+'/row_sing'+'.txt')
	saveFile(MSE_valid_7,FileName+'/valid_MSE_0.7'+'.txt')
	saveFile(MSE_test_7,FileName+'/test_MSE_0.7'+'.txt')
	saveFile(MSE_valid_8,FileName+'/valid_MSE_0.8'+'.txt')
	saveFile(MSE_test_8,FileName+'/test_MSE_0.8'+'.txt')
	saveFile(MSE_valid_9,FileName+'/valid_MSE_0.9'+'.txt')
	saveFile(MSE_test_9,FileName+'/test_MSE_0.9'+'.txt')
	saveFile(MSE_vanilla_valid,FileName+'/valid_MSE_vanilla'+'.txt')
	saveFile(MSE_vanilla_test,FileName+'/test_MSE_vanilla'+'.txt')
	saveFile(rank_stability_7,FileName+'/rank_stability_7'+'.txt')
	saveFile(rank_stability_8,FileName+'/rank_stability_8'+'.txt')
	saveFile(rank_stability_9,FileName+'/rank_stability_9'+'.txt')
	saveFile(rank_vanilla,FileName+'/rank_vanilla'+'.txt')
	
	

def biasRemoval(ratings,outfile):

	pairUser = ratings.map(lambda l: (l[0],(l[1],float(l[2]))))
	print(ratings.count())
	userDict = pairUser.aggregateByKey([], lambda a,b: a + [b],lambda x, y: x+y).collectAsMap()
	userID = list(userDict.keys())
	with open('ratings_movies_amazon_compressed.dat', "w") as f:
		for i in range(0,len(userID)):
			t = userDict[userID[i]]
			l1 = [int(x[1]) for x in t]
			avgRating = np.mean(np.array(l1))
			for j in range(0,len(t)):
				st = [userID[i],userDict[userID[i]][j][0],userDict[userID[i]][j][1]- avgRating] 
				str1 = '::'.join(str(e) for e in st)
				f.write( '%s\n' %(str1))
	f.close()



# setup some parameters
BiasRemoval = 1
embedDim = 80
maxBags = 100
maxIter = 30


# read data
sc = SparkContext()
sqlContext = SQLContext(sc) 
FILENAME = 'ratings_Video_Games.csv'
data = sqlContext.read.text(FILENAME).rdd
ratings = data.map(lambda l: l.value.split(","))

# # Find a mapping to users and items
userMap = ratings.\
 		map(lambda x: x[0]).\
 		distinct().\
 		zipWithIndex().\
 		collectAsMap()

itemMap = ratings.\
 		map(lambda x: x[1]).\
 		distinct().\
 		zipWithIndex().\
 		collectAsMap()

numUsers = ratings.map(lambda l: l[0]).distinct().count()
numItems = ratings.map(lambda l: l[1]).distinct().count()

ratings = ratings.map(lambda l: (str(l[0]), str(l[1]), float(l[2])))
ratings = ratings.map(lambda l: (userMap[l[0]],itemMap[l[1]],l[2]))
ratings = selectedUserMostObserved(ratings,40,10)
numUsers = ratings.map(lambda l: l[0]).distinct().count()
numItems = ratings.map(lambda l: l[1]).distinct().count()

# create training and testing data
(pairTrain,pairTest) = createTrainData(ratings,0.95)
(pairTrain,pairValid) = createTrainData(pairTrain,0.90)

# execute vanilla ALS and stability selection ALS
rankEstimation(pairTrain,pairTest,pairValid,numUsers,numItems,maxBags,embedDim)



