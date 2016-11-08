from traitement_texte import *
from fractions import Fraction
import numpy as np
from save import *

def clean_text (fileName):
	articles = []
	contents = []
	f = open("%s" %(fileName), "r")
	for row in f.readlines():
		data = row.split('\t')
		temp = remove_stopwords(remove_accents(remove_digit(lower_string(remove_punct(data[1].strip('\n').strip('\r'))))))
		articles.append(data[0])
		contents.append(temp)
	f.close()
	return (articles, contents)

articlesArt, contentsArt = clean_text ("articles/ArtResult.txt")
articlesEconomy, contentsEconomy = clean_text ("articles/EconomyResult.txt")
articlesScience, contentsScience = clean_text ("articles/ScienceResult.txt")
articlesSport, contentsSport = clean_text ("articles/SportResult.txt")

# create dictionary for each category: Art, Economy, Science; Sport
def count_words_in_category (contents):
	words = {}
	for content in contents:
		tokens = content.split()
		for token in tokens:
			if token in words:
				words[token] += 1
			else:
				words[token] = 1
	return words
# create dictionary for each article
def count_words_in_article (contents):
	words = {}
	tokens = contents.split()
	for token in tokens:
		if token in words:
			words[token] += 1
		else:
			words[token] = 1
	return words

# calculate total words of a dictionary
def count_total_words_in_dict (dict):
	total = 0
	for word, count in dict.items():
		total += count
	return total

# calculate denominator
def cal_denominator (dicOneCategory, dictAllCategory):
	return count_total_words_in_dict (dicOneCategory) + len(dictAllCategory)

# calculate the probability of word: P(X|Ci)
def cal_prob_word (dicOneCategory, dictAllCategory):
	prob = {}
	for word in dictAllCategory.keys():
		if word in dicOneCategory:
			count = dicOneCategory[word]
		else:
			count = 0
		prob[word] = Fraction(count + 1, cal_denominator(dicOneCategory, dictAllCategory))
	return prob

def exponent (m, n):
	return m**n

def naiveBayesTrain (trainArt, trainEconomy, trainScience, trainSport):
	dicArt = count_words_in_category (trainArt)
	dicEconomy = count_words_in_category (trainEconomy)
	dicScience = count_words_in_category (trainScience)
	dicSport = count_words_in_category (trainSport)
	dictAllCategory = {}
	dictAllCategory.update(dicArt)
	dictAllCategory.update(dicEconomy)
	dictAllCategory.update(dicScience)
	dictAllCategory.update(dicSport)
	probDicArt = cal_prob_word (dicArt, dictAllCategory)
	probDicEconomy = cal_prob_word (dicEconomy, dictAllCategory)
	probDicScience = cal_prob_word (dicScience, dictAllCategory)
	probDicSport = cal_prob_word (dicSport, dictAllCategory)
	prob = (probDicArt, probDicEconomy, probDicScience, probDicSport)
	return prob

def uniqueTest (newArticle, dictAllCategory, probDic, x):
	p = Fraction(1,1)
	dict_newArticle = count_words_in_article (newArticle)
	for newArticleKey in dict_newArticle:
		if newArticleKey in dictAllCategory:
			prob = probDic[newArticleKey]
			p *= exponent(prob, dict_newArticle[newArticleKey])
	return p * x

def naiveBayesTest (testData, classificationModel):
	probDicArt = classificationModel[0]
	probDicEconomy = classificationModel[1]
	probDicScience = classificationModel[2]
	probDicSport = classificationModel[3]
	dictAllCategory = {}
	dictAllCategory.update(probDicArt)
	dictAllCategory.update(probDicEconomy)
	dictAllCategory.update(probDicScience)
	dictAllCategory.update(probDicSport)
	resultArt = 0
	resultEconomy = 0
	resultScience = 0
	resultSport = 0
	for newArticle in testData:
		p1 = uniqueTest(newArticle, dictAllCategory, probDicArt, Fraction(1,4))
		p2 = uniqueTest(newArticle, dictAllCategory, probDicEconomy, Fraction(1,4))
		p3 = uniqueTest(newArticle, dictAllCategory, probDicScience, Fraction(1,4))
		p4 = uniqueTest(newArticle, dictAllCategory, probDicSport, Fraction(1,4))
		maxP = max(p1,p2,p3,p4)
		if maxP == p1:
			resultArt += 1
		elif maxP == p2:
			resultEconomy += 1
		elif maxP == p3:
			resultScience += 1
		else:
			resultSport += 1
	return [resultArt, resultEconomy, resultScience, resultSport]

def k_fold_cross_validation (K, contents):
	train = [x for i, x in enumerate(contents) if i % K != 0]
	test = [x for i, x in enumerate(contents) if i % K == 0]
	return train, test

trainArt, testArt = k_fold_cross_validation (3, contentsArt)
trainEconomy, testEconomy = k_fold_cross_validation (3, contentsEconomy)
trainScience, testScience = k_fold_cross_validation (3, contentsScience)
trainSport, testSport = k_fold_cross_validation (3, contentsSport)

classificationModel = naiveBayesTrain (trainArt, trainEconomy, trainScience, trainSport)

testArtResult = naiveBayesTest (testArt, classificationModel)
testEconomyResult = naiveBayesTest (testEconomy, classificationModel)
testScienceResult = naiveBayesTest (testScience, classificationModel)
testSportResult = naiveBayesTest (testSport, classificationModel)

confusion_matrix = []
confusion_matrix.append(testArtResult)
confusion_matrix.append(testEconomyResult)
confusion_matrix.append(testScienceResult)
confusion_matrix.append(testSportResult)
confusion_matrix = np.asarray(confusion_matrix)

accuracy = (testArtResult[0]*1.0/len(testArt) + testEconomyResult[1]*1.0/len(testEconomy) + testScienceResult[2]*1.0/len(testScience) + testSportResult[3]*1.0/len(testSport))/4

print "Result : " 
print confusion_matrix
print "Accurate : "
print accuracy*100
print "VOIR LE RESULTAT DANS FICHIER NaiveBayes.xls"

lenX = len(contentsArt) + len(contentsEconomy) + len(contentsScience) + len(contentsSport)
lenXtrain = len(trainArt) + len(trainEconomy) + len(trainScience) + len(trainSport)
lenXtest = len(testArt) + len(testEconomy) + len(testScience) + len(testSport)

############ Ecrire resultat sur un fichier Excel ###############
export ("NaiveBayes.xlsx", confusion_matrix, lenX, lenXtrain, lenXtest, accuracy)


