import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
import scipy.stats as st

def load_embeddings(file_url, bin_format=False, limit=50000, pre_process=True):
    if not bin_format:
        model = KeyedVectors.load_word2vec_format(file_url, binary=bin_format, limit=limit)
        vocab_dict = list(model.vocab.keys())
    else:
        vocab_dict=[]
        model={}
        with open(file_url,"r") as F:
            count=0
            for line in F:
                #print(line)
                if count==limit:
                    break
                count+=1

                word,vec=line.split()[0],np.array( list(map(float,line.split()[-300:]) ),dtype=float)
                model[word]=vec
                vocab_dict.append(word)
				

    print("Loaded", len(vocab_dict), "words")
    if pre_process==True:
        vocab_dict = list(filter(lambda x: x.islower() and len(x)<20 and x.isalpha(), vocab_dict))
        print(len(vocab_dict), "words left after preprocessing")
    return model, vocab_dict


def get_gender_specific_words(word_list_url, embedding_limit, vocab_limit, embedding_full, vocab_full):
	with open(word_list_url, "rb") as F:
	    S0 = F.readlines()
	S0 = [i.decode("utf-8")[:-2].split(',') for i in S0]
	S0 = [item.strip(' ') for sublist in S0 for item in sublist]
	print("Loaded S0")
	X_train, y_train = [], []
	for i in vocab_limit:
	    X_train.append(embedding_limit[i])
	    if i in S0:
	        y_train.append(1)
	    else:
	        y_train.append(0)
	print("Built train set")
	vocab_full = list(set(vocab_full)-set(vocab_limit))
	X_test = []
	for i in vocab_full:
	    X_test.append(embedding_full[i])
	print("Built test set")
	X_train, y_train = np.array(X_train), np.array(y_train)
	X_test = np.array(X_test)
	print(X_train.shape, X_test.shape)

	skf = StratifiedKFold(n_splits=10)
	skf.get_n_splits(X_train, y_train)
	kfold_score = []
	for train_index, test_index in skf.split(X_train, y_train):
	    X_train_t, X_test_t = X_train[train_index], X_train[test_index]
	    y_train_t, y_test_t = y_train[train_index], y_train[test_index]	
	    clf = LinearSVC(C=1.0, class_weight="balanced")
	    clf.fit(X_train_t, y_train_t)
	    kfold_score.append(clf.score(X_test_t, y_test_t))
	print(np.mean(kfold_score), st.t.interval(0.95, len(kfold_score)-1, loc=np.mean(kfold_score), scale=st.sem(kfold_score)))

	clf = LinearSVC(C=1.0)
	clf.fit(X_train, y_train)
	print("Fit model")
	y_pred = clf.predict(X_test)
	print("Prediction complete")
	S1 = []
	for i,j in zip(vocab_full, y_pred):
	    if j==1:
	        S1.append(i)

	return S1
