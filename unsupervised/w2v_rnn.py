from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections,math,os,random,pickle,zipfile
from tempfile import gettempdir
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def drawProgressBar(percent, barLen = 50):
	sys.stdout.write("\r")
	progress = ""
	for i in range(barLen):
		if i<int(barLen * percent):
			progress += "="
		else:
			progress += " "
	sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
	sys.stdout.flush()

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

tokens = []
for i in range(len(lines)):
	splits = lines[i].split(' ')
	for j in range(len(splits)):
		tokens.append(splits[j])
unique_tokens = set(tokens)
lines_size = len(unique_tokens)
print(len(unique_tokens))

def build_dataset(words, n_words):
	"""Process raw inputs into a dataset."""
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

data, count, dictionary, reverse_dictionary = build_dataset(tokens,len(tokens))#lines,lines_size
del lines  # Hint to reduce memory.
"""
print('Most common words (+UNK)', count[:5])
print('Least common words (+UNK)',count[-5:])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
"""

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
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

batch_size = 1024
embedding_size = 128  # Dimension of the embedding vector. 32. lets try higher dims
skip_window = 8       # How many words to consider left and right.
num_skips = 4         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.    



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 32     # Random set of words to evaluate similarity on.
valid_window = 1024  # Only pick dev samples in the head of the distribution.
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
				tf.random_uniform([lines_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)

		# Construct the variables for the NCE loss
		nce_weights = tf.Variable(
				tf.truncated_normal([lines_size, embedding_size],
														stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([lines_size]))
	
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 1.0
	learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step,15000, 0.9, staircase= False)

	loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights,
										 biases=nce_biases,
										 labels=train_labels,
										 inputs=embed,
										 num_sampled=num_sampled,
										 num_classes=lines_size))

	# Construct the SGD optimizer using a learning rate of 1.0.
	optimizer = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step))

	# What if we use AdamOptimizer
	#optimizer = (tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step = global_step))
	
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
num_steps = 55001

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

pickle.dump(final_embeddings,open('128dimsw2v','wb'))

"""
def plot_with_labels(low_dim_embs, labels, filename):
	assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
	plt.figure(figsize=(20,20))  # in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(label,
								 xy=(x, y),
								 xytext=(5, 2),
								 textcoords='offset points',
								 ha='right',
								 va='bottom')

	plt.show()
	#plt.savefig(filename)try:
	# pylint: disable=g-import-not-at-top
try:
	# pylint: disable=g-import-not-at-top
	from sklearn.manifold import TSNE
	import matplotlib.pyplot as plt
	#%matplotlib inline
	print("final_embeddings size",final_embeddings.shape)
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
	plot_only = final_embeddings.shape[0]//100
	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
	labels = [reverse_dictionary[i] for i in xrange(plot_only)]
	plot_with_labels(low_dim_embs, labels, os.path.join(os.getcwd(), str("_"+str(num_steps)+".png")))

except ImportError as ex:
	print('Please install sklearn, matplotlib, and scipy to show embeddings.')
	print(ex)    
"""