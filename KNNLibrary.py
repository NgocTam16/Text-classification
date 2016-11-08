import numpy as np
import pylab as plt
from traitement_texte import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
'''
C = CountVectorizer(max_df = .5,min_df = 1)
X = C.fit_transform(contenu)
X = X.todense()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, label,test_size = .3,random_state = 1984)
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)

from sklearn.cross_validation import StratifiedKFold
folds = StratifiedKFold(Ytrain, n_folds = 5)

accuracy = np.zeros((5,10))
valeurs_possible = range(1,11)

i = 0
for train, test in folds:
	xtrain = Xtrain[train,]
	xtest = Xtrain[test,]
	ytrain = Ytrain[train]
	ytest = Ytrain[test]
	j = 0
	for k in valeurs_possible:
		model = KNeighborsClassifier(n_neighbors = k)
		model.fit(xtrain,ytrain)
		predictions = model.predict(xtest)
		accuracy[i,j] = (ytest == predictions).mean()
		j += 1
	i += 1

#print(accuracy)
#accuracy_moyenne = accuracy.mean(0)
accuracy_moyenne = np.mean(accuracy, axis=0)
#print(accuracy_moyenne)
#print(np.argmax(accuracy_moyenne))

plt.xlabel('k')
plt.ylabel('Accuracy')
plt.plot(valeurs_possible,accuracy_moyenne,linewidth = 2)
plt.title('K nearest neighbors')
plt.show()

meilleurK = valeurs_possible[np.argmax(accuracy_moyenne)]

#model = BernoulliNB()

model = KNeighborsClassifier(n_neighbors = meilleurK)
model.fit(Xtrain,Ytrain)
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
print ("VOIR LE RESULTAT DANS FICHIER K-NNLibray.xls")


############ Ecrire resultat sur un fichier Excel ###############
export ("K-NNLibrary.xlsx", confusion_matrix, len(X), len(Xtrain), len(Xtest), accuracy, precision_score, recall_score, f1_score)
'''
import os
file = "K-NNLibrary.xlsx"

#Mac OS/X:
os.system("open "+file)
'''
#Windows:
#os.system("start "+filename)

