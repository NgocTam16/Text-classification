import numpy as np
import pylab as plt
from traitement_texte import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from save import *
sys.getdefaultencoding()

label = []
titre = []
contenu = []
x = 0

def GetData(textName):
	f = open(textName,'rb')
	for i in f.readlines():
		data = i.decode('utf8').split('\t')
		label.append(x)
		titre.append(data[0])
		contenu.append(clean_texte(data[1].strip('\n')))
	f.close()

def clean_texte(t):	
	t = lower_string(t)
	t = replace_url(t)
	t = remove_punct(t)
	t = remove_digit(t)
	t = remove_stopwords(t)
	return t	
'''
GetData('articles/ArtResult.txt')
x += 1
GetData('articles/EconomyResult.txt')
x += 1
GetData('articles/ScienceResult.txt')
x += 1
GetData('articles/SportResult.txt')
'''
GetData('articles-total/ArtResult.txt')
x += 1
GetData('articles-total/EconomyResult.txt')
x += 1
GetData('articles-total/ScienceResult.txt')
x += 1
GetData('articles-total/SportResult.txt')

C = CountVectorizer(max_df = .5,min_df = 1)
X = C.fit_transform(contenu)
X = X.todense()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, label,test_size = .3,random_state = 1984)
print ("len(X) = %d" % (len(X)))
print ("len(Xtrain) = %d" % (len(Xtrain)))
print ("len(Xtest) = %d" % (len(Xtest)))

'''
model = BernoulliNB()
model.fit(Xtrain,Ytrain)
prediction = model.predict(Xtest)
'''
model = svm.SVC(kernel='linear', C=1)
model.fit(Xtrain, Ytrain)
prediction = model.predict(Xtest)

accuracy = (Ytest==prediction).mean()
precision_score = precision_score(Ytest, prediction, average="macro")
recall_score = recall_score(Ytest, prediction, average="macro")
f1_score = f1_score(Ytest, prediction, average="macro")

print ("accuracy %.3f" %accuracy)
print("precision_score %.3f" %precision_score)
print("recall_score %.3f" %recall_score)
print("f1_score %.3f" %f1_score)

confusion_matrix = confusion_matrix(Ytest, prediction, labels = [0,1,2,3])
print (confusion_matrix)
print ("VOIR LE RESULTAT DANS FICHIER SVMLibrary.xlsx")

############ Ecrire resultat sur un fichier Excel ###############
export ("SVMLibrary.xlsx", confusion_matrix, len(X), len(Xtrain), len(Xtest), accuracy, precision_score, recall_score, f1_score)
'''
import os
file = "SVMLibrary.xlsx"

#Mac OS/X:
os.system("open "+file)
'''
