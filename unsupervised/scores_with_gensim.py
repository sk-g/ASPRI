# requires gensim
# usage: PYTHONHASHSEED=0 python3 scores_with_gensim.py --iter=25 --window=1 --min_count=1 --size=8
# add extra arguements --true_paths and --fake_paths with corresponding text files
# if score of a path is not in the range of (min(score(true_path)),max(score(true_path))) then 
# the path is invalid/fake


import gensim,logging,argparse
from gensim.models import KeyedVectors
import os,sys,time,pickle,statistics,time
from language_model import drawProgressBar
import numpy as np
from gensim.models.word2vec import *
#logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--iter', type=int, default=5,
					   help='number of iterations for Word2Vec')
	parser.add_argument('--sg', type=int, default=1,
					   help='skipgram usage')
	parser.add_argument('--dump', type=int, default=0,
					   help='save model')
	parser.add_argument('--window', type=int, default=4,
					   help='window in word2vec')
	parser.add_argument('--size', type=int, default=32,
					   help='embedding dimension size')
	parser.add_argument('--min_count', type=int, default=1,
					   help='minimum word count to use')
	parser.add_argument('--model',type=str,default = 'word2vec',
						help = 'Choose between FastText and Word2Vec')
	parser.add_argument('--true_paths',type=str,default='11012018.txt',
						help='file containing valid paths')
	parser.add_argument('--fake_paths',type=str,default='11012018_f.txt',
						help='file containing fake paths')
	args = parser.parse_args()
	print(args)
	train(args)
def train(args):
	print("\nInitialize model with {}\n".format(args.iter))
	def build(iter = args.iter):
		true_file = str(args.true_paths)
		fake_file = str(args.fake_paths)
		sentences = list(set([i.strip() for i in open(true_file,'r').readlines()]))
		sentences = [i.split(' ') for i in sentences]
		ttl = len(sentences)


		sentences = LineSentence(open(true_file,'r'))
		model = gensim.models.Word2Vec(compute_loss = True, min_count=args.min_count,size = args.size,hs = 1, sg = args.sg,seed = 694,\
			window = args.window, negative = 0,iter = 1,workers=1)		
		
		model.build_vocab(sentences)
		model.train(sentences, compute_loss = False,total_examples=model.corpus_count, epochs=args.iter)
		

		true_scores,fake_scores = [],[]

		true_scores = np.array(model.score(sentences,ttl))# log likelihoods

		del sentences

		
		fake_sentences = [i.split(' ') for i in list(set([i.strip() for i in open(fake_file,'r').readlines()]))]
		ttl = len(fake_sentences)
		
		#fake_sentences = LineSentence(open(fake_file,'r'))
		fake_scores = np.array(model.score(fake_sentences,ttl)) # log likelihoods
		true_scores = true_scores.astype(np.float64)#cast to float64 type otherwise rounds off
		fake_scores = fake_scores.astype(np.float64)
		true_scores,fake_scores = np.exp(true_scores),np.exp(fake_scores)#exponential to give softmax probs. Not required
		del fake_sentences
		#print("\nmodel.doesnt_match(6939 7545 7545 7545 7545 4651)\n",model.wv.doesnt_match("6939 7545 7545 7545 7545 4651".split()))
		#print("Maximum,minimum log likelihood over all valid sentences = {},{}\n\
		#	Maximum,minimum log likelihood over all invalid sentences = {},{}".format(max(true_scores),min(true_scores),max(fake_scores),min(fake_scores)))
		model.save(str(args.size)+'_'+str(args.window)+'_'+str(args.iter))
		return true_scores,fake_scores

	def loadScores():

		true_scores = pickle.load(open('gensim'+os.sep+'true_scores','rb'))
		true_scores = true_scores.astype(np.float64)
		fake_scores = pickle.load(open('gensim'+os.sep+'fake_scores','rb'))
		fake_scores = fake_scores.astype(np.float64)
		return true_scores,fake_scores
	def predictions(true_scores,fake_scores):

		ts,fs = true_scores,fake_scores
		minimum_true = min(ts)
		maximum_true = max(ts)

		ttl = len(fs)
		correct, wrong = 0, 0
		for i in range(len(fs)):
			if fs[i] > maximum_true or fs[i] < minimum_true:
				correct += 1#because these are supposed to be fake
							#and their score is beyond correct range
			else:
				wrong += 1
		#print("Total:{}".format(correct+wrong))
		print("{}/{} Correct.\tAccuracy = {}\n".format(correct,wrong,100*correct/(correct+wrong)))

	true_scores,fake_scores = build()
	print("\n")

	predictions(true_scores,fake_scores)
	## sanity check by giving true scores only
	## accuracy should be 0 in this call
	#predictions(true_scores,true_scores)


	def dumpScores():

		pickle.dump(true_scores,open('gensim'+os.sep+'window4_true_scores_'+str(iter),'wb'))
		pickle.dump(fake_scores,open('gensim'+os.sep+'window4_fake_scores_'+str(iter),'wb'))
	if args.dump:
		dumpScores()

if __name__ == '__main__':
	main()