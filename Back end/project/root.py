#import regex
import re
import csv
import random
import operator

#start process_status
def processStatus(status):
        #process the status
        #Convert to lower case
        status = status.lower()
        status = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', status)
        status = re.sub('@[^\s]+', 'AT_USER', status)
        status = re.sub('[\s]+', ' ', status)
        status = status.strip('\'"')
        return status
#end

#initialize stopWords
stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
        #look for 2 or more repetitions of character and replace with the character itself
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)
#nd
#start getStopWordList
def getStopWordList(stopWordListFileName):
        #read the stopwords file and build a list
        stopWords = []
        stopWords.append('AT_USER')
        stopWords.append('URL')

        fp = open(stopWordListFileName, 'r')
        line = fp.readline()
        while line:
                word = line.strip()
                stopWords.append(word)
                line = fp.readline()
        fp.close()
        return stopWords
#end

#calculate the euclidian distance
def distance(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i]) ** 2
    return s

#*****finding the majority class*******
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

#start getfeatureVector
def getFeatureVector(status, stopWords):
        featureVector = {}
        #split status into words
        words = status.split()
        for w in words:
                #replace two or more with two ocurrences
                w = replaceTwoOrMore(w)
                #strip punctuation
                w = w.strip('\'"?,.')
                #check if the word starts with an alphabet
                val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
                #ignore if it is a stop word
                if(w in stopWords or val is None):
                        continue
                else:
                        w = w.lower()
                        if w in featureVector: featureVector[w] += 1
                        else: featureVector[w] = 1
        return featureVector
#end

#Read the status one by one and process it

st = open('stopwords.txt', 'r')
stopwords = getStopWordList('stopwords.txt')

inpStatus = csv.reader(open('with_all.csv','rb'), delimiter = ',', quotechar='"')
statuses = []
usermap = {}
userclass = {}
userclass1 = {}
totalbagofwords = set()

for row in inpStatus:
    break

TAKE = 200
cnt = 0
for row in inpStatus:
        userid = row[0]
        status = row[1]
        ext = row[2]
        neu = row[3]
        agr = row[4]
        con = row[5]
        opn = row[6]
        cnt += userid not in usermap
        if cnt > TAKE:
            break
        processedStatus = processStatus(status)
        featureVector = getFeatureVector(processedStatus, stopwords )

        if userid not in usermap:
                usermap[userid] = {}
        for word in featureVector:
            totalbagofwords.add(word)
            count = featureVector[word]
            if word in usermap[userid]: usermap[userid][word] += count
            else: usermap[userid][word] = count
        userclass[userid] = [int(float(opn)), int(float(con)), int(float(ext)), int(float(agr)), int(float(neu))]
        #userclass1[userid] = int(float(opn))
        #statuses.append((featureVector, ext, neu, agr, con, opn))

totalbagofwords = list(totalbagofwords)
bagcount = {}
training_feature_set = []
test_set = []
test_users = set()

#****to store the word count present in the bow******
fo = open("wordCount.txt","w+")

'''for each user, generate the bag of words
which is basically a dictionary with count
the statuses from each user are put into the dictionary
along with the word count'''
for row in inpStatus:
        userid = row[0]
        test_users.add(userid)
        status = row[1]
        ext = row[2]
        neu = row[3]
        agr = row[4]
        con = row[5]
        opn = row[6]
        
        processedStatus = processStatus(status)
        featureVector = getFeatureVector(processedStatus, stopwords)

        fo.write(str(featureVector.values())+"\n")

        if userid not in usermap:
                usermap[userid] = {}
        for word in featureVector:
            if word not in bagcount: bagcount[word] = 0
            bagcount[word] += 1
            count = featureVector[word]
            if word in usermap[userid]: usermap[userid][word] += count
            else: usermap[userid][word] = count
        userclass[userid] = [int(float(opn)), int(float(con)), int(float(ext)), int(float(agr)), int(float(neu))]
        #userclass1[userid] = int(float(opn))
        #statuses.append((featureVector, ext, neu, agr, con, opn))

fo.close()

maxcount = {}
for i in usermap:
    for j in usermap[i]:
        if j not in maxcount:
            maxcount[j] = 0
        maxcount[j] = max(maxcount[j], usermap[i][j])

'''separate test sets and training set from the data universe
TAKE amount of training feature sets are generated
remaining goes for testing'''
for i in usermap:
    data = usermap[i]

    feature = [.5 + .5 * data[j] / (0 if j not in maxcount else maxcount[j])  if j in data else 0 for j in totalbagofwords]
    #feature = [1 * data[j]  if j in data else 0 for j in totalbagofwords]
    
    if i in test_users:
        test_set.append([feature, userclass[i]])

    else:
        training_feature_set.append([feature, userclass[i]])

#print test_set[0][1]

#**********writing the feature vector in file******************
fo1 = open("testFeature.txt","w+")
for i,j in test_set:
    for k in i:
        fo1.write("%s " %k)
    for m in j:
        fo1.write("%s " %m)
    fo1.write("0\n")
fo1.close()

fo2 = open("trainingFeature.txt","w+")
for i,j in training_feature_set:
    for k in i:
        fo2.write("%s " %k) 
    for m in j:
        fo2.write("%s " %m)
    fo2.write("0\n")
fo2.close()
#end


misclassified = 0

'''implement knn algorithm
n nearest points are calculated, the final class is the majority
of the classes of those n points'''

for N in range(3, 50):
    misclassified = 0
    for feature, cls in test_set:
        dist = [1e15] * N
        classification = [0] * N
        #*******Calculate the euclidian distance of each test set with every training set***********
        for f2, cls2 in training_feature_set:
            dn = distance(feature, f2) 
            for j in range(N):
                if dn < dist[j]:
                    dist[j] = dn
                    classification[j] = cls2
                
                    break
        #final = sum(classification) / N
        final = getResponse(classification)
        
        #print 'original class ' + str(cls) + ' classified as ' + str(final)
        misclassified += cls != final
    print "k =", str(N) + ' ' + ' correct prediction ='+ str(50-misclassified) + ' accuracy ' , (1 - misclassified / 50.0)*100 , '%'
#end loop
