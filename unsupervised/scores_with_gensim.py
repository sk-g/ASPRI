import gensim,logging
from gensim.models import KeyedVectors
import os,sys,time,pickle,statistics
from language_model import drawProgressBar
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def build():
	sentences = [i.strip() for i in open('11012018.txt','r').readlines()]
	sentences = list(set(sentences))
	sentences = [i.split(' ') for i in sentences]
	
	model = gensim.models.Word2Vec(sentences, min_count=1,size = 128,hs = 1, sg = 1,seed = 694,window = 16, negative = 0,iter = 5)
	#f = open('gensim scores.txt','a')
	#sys.stdout = open('gensim scores.txt','w')
	#print([(i,gensim.models.deprecated.word2vec.score_sentence_sg(model,i)) for i in sentences])
	ttl = len(sentences)
	true_scores = []
	print("\nScoring valid paths ...\n")
	for i in range(ttl):
		#f.write(' '+str(i)+'\t' + str(gensim.models.deprecated.word2vec.score_sentence_sg(model,sentences[i])))
		#true_scores.append(gensim.models.deprecated.word2vec.score_sentence_sg(model,sentences[i]))
		true_scores.append(model.score(sentences[i]))
		drawProgressBar(i/ttl)
	del sentences
	fake_sentences = list(set([i.strip() for i in open('11012018_f.txt','r').readlines()]))
	print("\nScoring invalid paths ...\n")
	fake_scores = []
	ttl = len(fake_sentences)
	for i in range(ttl):
		#f.write(str(i)+'\t' + str(gensim.models.deprecated.word2vec.score_sentence_sg(model,fake_sentences[i]))+'\n')
		#fake_scores.append(gensim.models.deprecated.word2vec.score_sentence_sg(model,fake_sentences[i]))
		fake_scores.append(model.score(fake_sentences[i]))
		drawProgressBar(i/ttl)
	del fake_sentences
	return true_scores,fake_scores

def loadScores():

	true_scores = pickle.load(open('gensim'+os.sep+'true_scores','rb'))
	fake_scores = pickle.load(open('gensim'+os.sep+'fake_scores','rb'))	
	return true_scores,fake_scores
def predictions(true_scores,fake_scores):
	#ts = [abs(i) for i in true_scores]
	#fs = [abs(i) for i in fake_scores]
	ts,fs = true_scores,fake_scores
	minimum_true = min(ts)
	maximum_true = max(ts)
	#print("\nvalid ones",minimum_true,maximum_true)
	#print("\ninvalid ones",min(fs),max(fs))
	ttl = len(fs)
	correct, wrong = 0, 0
	for i in range(len(fs)):
		if fs[i] > maximum_true or fs[i] < minimum_true:
			correct += 1#because these are supposed to be fake
						#and their score is beyond correct range
		else:
			wrong += 1
	print("Total:{}".format(correct+wrong))
	print("Correct = {}\nWrong = {}\nAccuracy = {}\n".format(correct,wrong,100*correct/(correct+wrong)))

true_scores,fake_scores = build()
print(fake_scores[:100])
#true_scores,fake_scores = loadScores()
predictions(true_scores,fake_scores)
#predictions(true_scores,true_scores)


def dumpScores():

	pickle.dump(true_scores,open('gensim'+os.sep+'true_scores','wb'))
	pickle.dump(fake_scores,open('gensim'+os.sep+'fake_scores','wb'))
	# skipping list comprehension to get progress bar
	#print([(i,gensim.models.deprecated.word2vec.score_sentence_sg(model,i)) for i in fake_sentences])
	#sys.stdout = sys.__stdout__