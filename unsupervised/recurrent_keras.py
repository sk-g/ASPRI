import os,keras
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU,LSTM
import numpy as np
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
paths = [(i.strip()) for i in open('11012018.txt','r')]
paths = list(set(paths))
fake_paths =[i.strip() for i in open('11012018_f.txt','r')]
tokenizer = Tokenizer(lower = False)
tokenizer.fit_on_texts(paths)
word2id = tokenizer.word_index
id2word = {idx: word for (word, idx) in word2id.items()}

maxSequenceLength = 30
encoded = tokenizer.texts_to_sequences(paths)
encoded = keras.preprocessing.sequence.pad_sequences(encoded,maxlen = (maxSequenceLength + 1), padding = 'post', truncating = 'post')
encoded_test = tokenizer.texts_to_sequences(fake_paths)
encoded_test = keras.preprocessing.sequence.pad_sequences(encoded_test,maxlen = (maxSequenceLength + 1), padding = 'post', truncating = 'post')
t = encoded[1000]
encoded = encoded[:100]
id2word[0] = 'END'
word2id['END'] = 0
print(encoded.shape)
vocabularySize = len(id2word)

print('Building training model...')

# Remember that in libraries like Keras/Tensorflow, you only need to implement the forward pass.
# Here we show how to do that for our model.

# Define the shape of the inputs batchSize x (maxSequenceLength + 1).
words = keras.layers.Input(batch_shape=(None, maxSequenceLength), name = "input")

# Build a matrix of size vocabularySize x 300 where each row corresponds to a "word embedding" vector.
# This layer will convert replace each word-id with a word-vector of size 300.
embeddings = keras.layers.embeddings.Embedding(vocabularySize, 32, name = "embeddings")(words)

# Pass the word-vectors to the LSTM layer.
# We are setting the hidden-state size to 512.
# The output will be batchSize x maxSequenceLength x hiddenStateSize
hiddenStates = keras.layers.GRU(64, return_sequences = True, 
									  input_shape=(maxSequenceLength, 32), name = "rnn")(embeddings)

# Apply a linear (Dense) layer of size 512 x 256 to the outputs of the LSTM at each time step.
denseOutput = TimeDistributed(keras.layers.Dense(vocabularySize), name = "linear")(hiddenStates)
predictions = TimeDistributed(keras.layers.Activation("softmax"), name = "softmax")(denseOutput)                                      

# Build the computational graph by specifying the input, and output of the network.
model = keras.models.Model(input = words, output = predictions)

# Compile the graph so that we have a way to compute gradients.
# We also specify here the type of optimization to perform. For Recurrent Neural Networks, a type of
# optimization called RMSprop is preferred instead of the standard SGD udpates.
model.compile(loss='sparse_categorical_crossentropy', optimizer = keras.optimizers.RMSprop(lr = 0.001))

print(model.summary()) # Convenient function to see details about the network model.


# Sample 10 inputs from the training data and verify everything works.
sample_inputs = encoded[0:10,:-1]
sample_outputs = model.predict(sample_inputs)
print('input size', sample_inputs.shape)
print('output size', sample_outputs.shape)

inputData = encoded[:, :-1]  # words 1, 2, 3, ... , (n-1)
outputData = encoded[:, 1:]  # words 2, 3, 4, ... , (n)

# We have to add an extra dimension if using "sparse_categorical_crossentropy".
# Sparse is always better if you want to save memory. Only store the non-zeros.
# Read here: https://keras.io/objectives/
outputLabels = np.expand_dims(outputData, -1)

# The labels have to be equal size to the outputs of the network if using "categorical_crossentropy" in Keras.
# we have to encode the labels as one-hot vectors. There is a function in Keras to do this.
# from keras.utils.np_utils import to_categorical
# print('Converting labels to one-hot encodings..')
# outputLabels = to_categorical(outputData, nb_classes = vocabularySize)
# outputLabels = np.reshape(outputLabels, (outputData.shape[0], outputData.shape[1], vocabularySize))
# print('Finishing converting labels to one-hot encodings')
# I commented out and abandoned this because it required too much memory!

checkpointer = keras.callbacks.ModelCheckpoint(filepath="my_weights.hdf5", save_weights_only = True, \
											   save_best_only = True, monitor = 'loss')
model.fit(inputData, outputLabels, batch_size = 512, epochs= 1, callbacks = [checkpointer])

# We could also go batch by batch ourselves, however the above function worked well so let's not go this way.
# trainSize = inputData.shape[0]
# batchSize = 100
# nBatches =  trainSize / batchSize
# for b in range(0, nBatches):
	 # Build the batch inputs, and batch labels.
#    batchInputs = np.zeros((batchSize, inputData.shape[1]))
#    batchLabels = np.zeros((batchSize, inputData.shape[1], vocabularySize))
#    for bi in range(0, batchSize):
#        rand_int = random.randint(0, trainSize - 1)
#        batchInputs[bi, :] = inputData[rand_int, :]
#        for s in range(0, inputData.shape[1]):
#            batchLabels[bi, s, outputData[rand_int, s]] = 1
#    
#    model.train_on_batch(batchInputs, batchLabels)

model.save_weights('my_language_model.hdf5')
"""
rnn = Sequential()

lexicon_size = len(tokenizer.word_index)
n_embedding_nodes = 32
n_hidden_nodes = 64
batch_size = 1
n_timesteps = None

#word embedding layer
embedding_layer = Embedding(batch_input_shape=(batch_size, n_timesteps),
														input_dim=lexicon_size + 1, #add 1 because word indices start at 1, not 0
														output_dim=n_embedding_nodes, 
														mask_zero=True)
rnn.add(embedding_layer)
#recurrent layers (GRU)
#recurrent_layer1 = GRU(output_dim=n_hidden_nodes,
#											 return_sequences=True, 
#											 stateful=True)
#rnn.add(recurrent_layer1)

recurrent_layer2 = GRU(output_dim=n_hidden_nodes,
											 return_sequences=True, 
											 stateful=True)
rnn.add(recurrent_layer2)

#prediction (softmax) layer
pred_layer = TimeDistributed(Dense(lexicon_size + 1, #add 1 because word indices start at 1, not 0
																	 activation="softmax"))
rnn.add(pred_layer)

#select optimizer and compile
rnn.compile(loss="sparse_categorical_crossentropy", 
						optimizer='adam')
#rnn.summary()
def train_epoch(paths):
	losses = []
	prev_eos = None
	encoded_path = encoded
	for path in encoded[:10]:
			path = np.array(path)
			if prev_eos:
					path = np.insert(path,0,prev_eos)
			sent_x = path[None,:-1]
			sent_y = path[None,1:,None]
			loss = rnn.train_on_batch(x=sent_x,y=sent_y)
			losses.append(loss)
			prev_eos = path[-1]
	rnn.reset_states()
	loss = np.mean(loss)
	return loss
n_epochs = 10
print( "Training RNN on", len(paths), "paths for", n_epochs, "epochs...")
#for epoch in range(n_epochs):
		#loss = train_epoch(paths)
		#print("epoch {} loss: {:.3f}".format(epoch + 1, loss))

sent = np.array(encoded[101])[None,:]
p = rnn.predict_on_batch(sent)
p = np.argmax(p,axis = 2)
print(sent.shape,p.shape)

def perplexity(y_true, y_pred):
	cross_entropy = K.categorical_crossentropy(y_true, y_pred)
	perplexity = K.pow(2.0, cross_entropy)
	return perplexity
#print(perplexity(sent,p))
"""
