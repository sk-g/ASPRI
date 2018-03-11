from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections,math,os,random,pickle,zipfile,sys
from tempfile import gettempdir
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
from language_model import drawProgressBar

if os.name != 'posix':
	import matplotlib.pyplot
	import seaborn as sns
	import pylab as pl
	f = open(r'..\data\PCH\paths\11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]
else:
	f = open('../data/PCH/paths/11012018.txt')
	lines = f.readlines()
	lines = [i.strip() for i in lines]



def addTokens(arr):
	# function to add
	# start token and
	# end token to all sentences
	# in array
	print("\nAdding start and end tokens\n")
	ttl = len(arr)
	for i in range(ttl):
		temp = arr[i].split(' ')
		temp.insert(0,'START')
		temp.append('END')
		temp = ' '.join([word for word in temp])
		arr[i] = temp
		drawProgressBar(i/ttl)
	return arr
#lines = addTokens(lines)
tokens = []
for i in range(len(lines)):
	splits = lines[i].split(' ')
	for j in range(len(splits)):
		tokens.append(splits[j])
unique_tokens = set(tokens)
unique_words = len(unique_tokens)
print("Number of unique ASes = {}".format(len(unique_tokens)-1))
def build_dataset(words, n_words):
	print("""Process raw inputs into a dataset.""")
	count = [['UNK', -1]]
	#count = [[]]
	count.extend(collections.Counter(words).most_common(n_words - 1))
	dictionary = dict()
	#print(count)
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		index = dictionary.get(word, 0)
		#print(word,index)
		if index == 0:  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(tokens,len(tokens))
#unique_words = len(lines)
del lines  # Hint to reduce memory.
"""
print('Most common words (+UNK)', count[:5])
print('Least common words (+UNK)',count[-5:])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
"""
pickle.dump(dictionary,open('tokenizedDic','wb'))
pickle.dump(count,open('tokenizedCount','wb'))
data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
	#print("\n Generating Batches \n")
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)
	if data_index + span > len(data):
		data_index = 0
	buffer.extend(data[data_index:data_index + span])
	data_index += span
	for i in range(batch_size // num_skips):
		context_words = [w for w in range(span) if w != skip_window]
		words_to_use = random.sample(context_words, num_skips)
		for j, context_word in enumerate(words_to_use):
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[context_word]
		if data_index == len(data):
			#buffer[:] = data[:span]
			for word in data[:span]:
				buffer.append(word)
			data_index = span
		else:
			buffer.append(data[data_index])
			data_index += 1
	# Backtrack a little bit to avoid skipping words in the end of a batch
	data_index = (data_index + len(data) - span) % len(data)
	return batch, labels

# Step 4: Build and train a skip-gram model.

batch_size = 256
embedding_size = 64  # Dimension of the embedding vector. 32. lets try higher dims
skip_window = 4       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 30      # Number of negative examples to sample.    



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 512  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()

with graph.as_default():

	# Input data.
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	# Ops and variables pinned to the CPU because of missing GPU implementation
	with tf.device('/gpu:0'):
		# Look up embeddings for inputs.
		embeddings = tf.Variable(
				tf.random_uniform([unique_words, embedding_size], -1.0, 1.0))/np.sqrt(embedding_size)#2.6k,embb_dim
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)

		# Construct the variables for the NCE loss
		nce_weights = tf.Variable(
				tf.truncated_normal([unique_words, embedding_size],
														stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([unique_words]))
	
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 1.0
	learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step,15000, 0.9, staircase= False)

	loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights,
										 biases=nce_biases,
										 labels=train_labels,
										 inputs=embed,
										 num_sampled=num_sampled,
										 num_classes=unique_words))

	# Construct the SGD optimizer using a learning rate of 1.0.
	#optimizer = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step))

	# What if we use AdamOptimizer
	optimizer = (tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step))
	
	# Compute the cosine similarity between minibatch examples and all embeddings.
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),
	 1, keepdims=True)) #keep_dims is deprecated use keepdims instead
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(
			normalized_embeddings, valid_dataset)
	similarity = tf.matmul(
			valid_embeddings, normalized_embeddings, transpose_b=True)
	#lr_print = tf.eval(learning_rate,[learning_rate])
	# Add variable initializer.
	init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 155001

## SGD ##

import time
start = time.time()
with tf.Session(graph=graph) as session:
	# We must initialize all variables before we use them.
	init.run()
	print('Initialized')

	average_loss = 0
	for step in xrange(num_steps):
		batch_inputs, batch_labels = generate_batch(
				batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		# We perform one update step by evaluating the optimizer op (including it
		# in the list of returned values for session.run()
		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val
		if step % 1000 == 0:
			if step > 0:
				average_loss /= 1000
			__ = float(learning_rate.eval())
			average_loss_ = float(average_loss)
			print("\nAverage loss at step {} : {:.3f}, learning rate = {:.3f}".format(step,average_loss_,__))
			#print('Average loss at step ', step, ': ', average_loss,'learning rate: ',__)
			average_loss = 0
	final_embeddings = normalized_embeddings.eval()
end = time.time()
seconds = end - start
minutes = seconds//60
seconds = seconds % 60
hours = 0
if minutes > 60:
	hours = minutes//60
	minutes = minutes%60
print("time taken for running the notebook:\n {0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))

pickle.dump(final_embeddings,open('final_wordEmbeddings','wb'))