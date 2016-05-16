from sklearn import svm
from sklearn import decomposition
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data = np.loadtxt("../project/data_with_all_label/trainingFeature.txt", delimiter =' ')
X = data[:, 0:12724]
y = data[:, 12724:12729]
y = y.transpose()

testdata = np.loadtxt("../project/data_with_all_label/testFeature.txt", delimiter =' ')
Xt = testdata[:, 0:12724]
yt = testdata[:, 12724:12729]
yt = yt.transpose()


#X = SelectKBest(chi2, k=1500).fit_transform(X, y)
#Xt = SelectKBest(chi2, k=1500).transform(Xt)

pca = decomposition.PCA(n_components = 10).fit(X)
X = pca.transform(X)


testpca = decomposition.PCA(n_components = 10).fit(Xt)
Xt = testpca.transform(Xt)

'''
model = svm.SVC(kernel='linear', C=1, gamma=100)

#model = svm.SVR()
model.fit(X,y)


predicted = model.predict(Xt)

clf = DecisionTreeClassifier() 
clf.fit(X,y)
predicted = clf.predict(Xt)
'''

def svm_Model(X, y, Xt, yt):
	model = OneVsRestClassifier(svm.LinearSVC(C=1)).fit(X,y)

	predicted = model.predict(Xt)

	acc = model.score(Xt,yt)

	#print 'Actual output = \n',yt
	#print 'Predicted output = \n',predicted

	print 'Accuracy = ', acc* 100 ,'%'

print "\n*******For Oppenness*********\n"
svm_Model(X, y[0], Xt, yt[0])

print "\n*******For Concientiousness*********\n"
svm_Model(X, y[1], Xt, yt[1])

print "\n*******For Extroversion*********\n"
svm_Model(X, y[2], Xt, yt[2])

print "\n*******For Agreeableness*********\n"
svm_Model(X, y[3], Xt, yt[3])

print "\n*******For Neuroticism*********\n"
svm_Model(X, y[4], Xt, yt[4])
