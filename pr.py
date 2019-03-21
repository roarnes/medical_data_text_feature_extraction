import numpy as np
# print(numpy.__path__)

#String for punctuation removal
import string

from collections import Counter

#Sastrawi Library for stemmatization in Bahasa
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#Stastrawi library for stop words removal in Bahasa
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#NLTK library for tokenization
import nltk
from nltk.tokenize import WordPunctTokenizer

#math library for calculation
import math

#calculating cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial


# OPENING FILE
text_file1 = open("trainingdata.txt", "r")
text_file2 = open("label.txt", "r")

#Training Data storing
lines = text_file1.read()
lines = [s for s in lines.split('"') if s.strip() != '']

# print ("Number of training data: " + str(len(lines)))
text_file1.close()

training_data = []
for item in lines:
    training_data.append(item.lower())

print("Input data: " , lines[2])
print("Case folding result: ", training_data[2])

#Labels storing
labels = text_file2.read().lower()
labels = [s for s in labels.split('"') if s.strip() != '']

# print ("Number of label data: " + str(len(labels)))
text_file2.close()

print("Disease: ", labels[2])

#PRE-PROCESSING

#STEMMATIZATION
# create stemmer
s_factory = StemmerFactory()
stemmer = s_factory.create_stemmer()

#stemming process
training_data_stemmed = []
for item in training_data:
	item = stemmer.stem(item)
	training_data_stemmed.append(item)

print("Stemming result: ", training_data_stemmed[2])

#STOP WORDS REMOVAL
#create remover
sw_factory = StopWordRemoverFactory()
stopwords = sw_factory.create_stop_word_remover()

#stop words removal process
training_data_sw = []
for word in training_data_stemmed:
	word = stopwords.remove(word)
	training_data_sw.append(word)
print("Stop word removal result: ", training_data_sw[2])	

#TOKENIZATION
#create tokenizer
tok = WordPunctTokenizer();

#tokenization process
training_data_tokenized = []
for word in training_data_sw:
	training_data_tokenized.append(tok.tokenize(word))
	
print("Tokenization result: ", training_data_tokenized[2])


#STORING TERMS
terms = []
for i in range(0, len(training_data_tokenized)):
	for word in training_data_tokenized[i]:
		if word not in terms:
			terms.append(word)

print(len(training_data_tokenized)) #93
print(len(terms)) #376

#TERM FREQUENCY
#assigning freq to [no. of training_data_tokenized][no. of terms]
#TFi,j
freq = [[0.0 for word in range(len(terms))] for line in range(len(training_data_tokenized))]

freq_all = [[0] for word in range(len(terms))]

for i in range(0, len(training_data_tokenized)):
	counts = Counter(training_data_tokenized[i])
	# print("sentence length: ", len(training_data_tokenized[i]))
	for j in range(0, len(terms)):
		for k in range (0, len(list(counts.values()))):
			if terms[j] == list(counts.keys())[k]:
				#freq = (Number of times term j appears in doc i) / (Total number of terms in the document)
				freq[i][j] = (float(list(counts.values())[k] / len(training_data_tokenized[i]) ))
				freq_all[j] += [list(counts.values())[k]]

# print(terms)
print(freq[2])
print(freq[2][10])
print("Occurences of word ", terms[2], ": " , sum(freq_all[2]))

#INVERSE DOCUMENT FREQUENCY
idf = [ [ 0.0 for word in range(len(terms)) ] for line in range(len(terms))]

for x in range (0, len(idf)):
	#idf = log_10(Total number of documents / Number of documents with term x in it)
	idf[x][x] = math.log10( (len(training_data_tokenized)) / (sum(freq_all[x])) )

# print(idf)
# print(freq[0][0] * idf[0][0])

#WEIGHT // TF-IDF
#Wi,j = TFi,j x IDFj
weight = [[ 0.0 for word in range(len(idf))] for line in range(len(freq))]

#matrix multiplication of freq[][] and idf[]
#iterate through rows of freq
# for i in range(0,len(freq)):
#    # iterate through columns of idf
# 	for j in range(0,len(idf[0])):
#        # iterate through rows of idf
# 		for k in range(0,len(idf)):
# 			# print(i, j , k)
# 			# print(weight[i][j])
# 			xx = freq[i][k] * idf[k][j]
# 			# print(xx)
# 			# print(freq[i][k] , idf[k][j])
# 			weight[i][j] += xx
# 			# print(weight[i][j])

weight = [[sum(a*b for a,b in zip(freq_row,idf_col)) for idf_col in zip(*idf)] for freq_row in freq]
			
print("Weight: " , weight[1])

###############

#GETTING NEW DATA
text_file3 = open("inputdata.txt", "r")

#New data storing
new_data = text_file3.read().lower()
new_data = [s for s in new_data.split('"') if s.strip() != '']

print ("Number of label data: " + str(len(new_data)))
text_file3.close()

#Pre-process new data
#stemming process
new_data_stemmed = []
for item in new_data:
	item = stemmer.stem(item)
	new_data_stemmed.append(item)

print("Stemming result: ", new_data_stemmed[0])

#STOP WORDS REMOVAL

#stop words removal process
new_data_sw = []
for word in new_data_stemmed:
	word = stopwords.remove(word)
	new_data_sw.append(word)
print("Stop word removal result: ", new_data_sw[0])	

#TOKENIZATION
#tokenization process
new_data_tokenized = []
for word in new_data_sw:
	new_data_tokenized.append(tok.tokenize(word))

print("Tokenization result: ", new_data_tokenized[0])

#NEW DATA

#TERM FREQUENCY
#assigning freq to [no. of training_data_tokenized][no. of terms]
#TFi,j
freq_new = [[0.0 for word in range(len(terms))] for line in range(len(new_data_tokenized))]

freq_all_new = [[0] for word in range(len(terms))]

for i in range(0, len(new_data_tokenized)):
	countss = Counter(new_data_tokenized[i])
	# print("sentence length: ", len(training_data_tokenized[i]))
	for j in range(0, len(terms)):
		for k in range (0, len(list(countss.values()))):
			if terms[j] == list(countss.keys())[k]:
				#freq = (Number of times term j appears in doc i) / (Total number of terms in the document)
				freq_new[i][j] = (float(list(countss.values())[k] / len(new_data_tokenized[i]) ))
				freq_all_new[j] += [list(countss.values())[k]]

# print(terms)
print(freq_new[2])
print(freq_new[2][10])
print("Occurences of word ", terms[2], " in new data: " , sum(freq_all_new[2]))

#INVERSE DOCUMENT FREQUENCY
idf_new = [ [ 0.0 for word in range(len(terms)) ] for line in range(len(terms))]

for x in range (0, len(idf_new)):
	#idf = log_10(Total number of documents / Number of documents with term x in it)
	if (sum(freq_all_new[x]) != 0):
		idf_new[x][x] = math.log10( (len(new_data_tokenized)) / (sum(freq_all_new[x])) )


#WEIGHT
weight_new = [ [ 0.0 for word in range(len(idf_new)) ] for line in range(len(freq_new))]

weight_new = [[sum(a*b for a,b in zip(freq_new_row, idf_new_col)) for idf_new_col in zip(*idf_new)] for freq_new_row in freq_new]
		
print("Weight of new data: " , weight_new[1])
print()

##################
#SIMILARITY CALCULATION

#MANHATTAN

man_similarity = 0.0
label_index = [0] * len(weight_new)

# for x in range(1,10):
# 	man_similarity = abs(weight - weight_new)

temp = 0.0
first_occurence = True
print(len(weight_new))
print(len(weight))
print(len(weight[0]))
print(len(labels))

for i in range (0, len(weight_new)):
	for j in range (0, len(weight)):
		for k in range (0, len(weight[0])):
			# print(weight_new[i][k], weight[j][k])
			#only calculate if the words in both matrices are the same
			if weight[j][k] > 0.0 and weight_new[i][k] > 0.0:
				temp += abs(weight_new[i][k] - weight[j][k])
				if first_occurence:
					man_similarity = temp
					first_occurence = False
		# print('temp: ', temp)
		#find minimum distance	
		if temp < man_similarity:
			man_similarity = temp
			label_index[i] = j
		# print('similarity',  man_similarity)
		temp = 0.0
		first_occurence = True

print(label_index)

for i in range (0, len(new_data_tokenized)):
	print('Symptom: ', new_data_tokenized[i])
	print('Label: ', labels[label_index[i]])

#COSINE SIMILARITY

cos_similarity = 0.0
temp2 = [0.0]
label_index2 = [0]* len(weight_new)

first_occurence = True

for i in range (0, len(weight_new)):
	for j in range (0, len(weight)):
		temp2 = 1 - spatial.distance.cosine(weight_new[i], weight[j])
		# print('temp: ', temp2)
		if first_occurence and temp2 != 0.0:
			cos_similarity = temp2
			first_occurence = False
		if temp2 > cos_similarity:
			cos_similarity = temp2
			label_index2[i] = j
		# print('similarity: ', cos_similarity)
	temp2 = 0.0
	first_occurence = True

print(label_index2)

for i in range (0, len(new_data_tokenized)):
	print('Symptom: ', new_data_tokenized[i])
	print('Label: ', labels[label_index2[i]])