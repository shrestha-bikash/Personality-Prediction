from sklearn.feature_selection import SelectKBest, chi2
from numpy import loadtxt, dstack,hstack, append, concatenate
import operator

'''list = [[[1,2,3,4,5,6,7,8],0],[[0,9,8,7,6,5,4,3,2],9]]
fo = open('check1.txt','w')
for i,j in list:
for k in i:
fo.write("%s " %k)
fo.write("%s" %j)
fo.write("\n")

fo.close()'''

#calculate the distance
def distance(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i]) ** 2
        return s

def getResponse(classification):
    classvote = {}
    for x in range(len(classification)):
        response = classification[x]
        if response in classvote:
            classvote[response] += 1
        else:
            classvote[response] = 1
    sortedvotes = sorted(classvote.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedvotes[0][0]


data = loadtxt('opn_data/trainingFeature.txt', delimiter=' ')
X = data[:, 0:12724]
y = data[:, 12724]


testdata = loadtxt("opn_data/testFeature.txt", delimiter =' ')
Xt = testdata[:, 0:12724]
yt = testdata[:, 12724]

X_train_new = SelectKBest(chi2, k=1500).fit_transform(X, y)
Xt_test_new = SelectKBest(chi2, k=1500).fit_transform(Xt, yt)

print yt
Xt_test=[]
yt_ctr = 0
for row in Xt_test_new:
    Xt_test.append([row,yt[yt_ctr]])
    yt_ctr += 1 

X_train = []
yt_ctr = 0
for row in X_train_new:
    X_train.append([row,y[yt_ctr]])
    yt_ctr += 1 

print Xt_test_new.shape
print len(Xt_test_new)
#print X_train_new.shape

#print Xt_test

for N in range(1, 50):
    misclassified = 0
    for onerow in Xt_test:
        feature = onerow[0]
        cls = onerow[1]
        dist = [1e15] * N
        classification = [0] * N
        #*******Calculate the euclidian distance of each test set with every training set***********
        for anotherrow in X_train:
            f2,cls2 = anotherrow[0],anotherrow[1]
            dn = distance(feature, f2)  #*****distance calculation*****
            for j in range(N):
                if dn < dist[j]:
                    dist[j] = dn
                    classification[j] = cls2
                    break
        #final = sum(classification) / N
        final = getResponse(classification)
        #print final
        #print 'original class ' + str(cls) + ' classified as ' + str(final)
        misclassified += cls != final
    print "k =", str(N) + ' ' + ' correct prediction ='+ str(50-misclassified) + ' accuracy ' , (1 - misclassified / 50.0)*100 , '%'
#end loop
