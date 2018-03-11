import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
files = os.listdir()

entropy = pickle.load(open('stacked_gru_entropy','rb'))
mse = pickle.load(open('stacked_gru_mse','rb'))
msle = pickle.load(open('stacked_gru_msle','rb'))
kl = pickle.load(open('stacked_gru_kl','rb'))
sqh = pickle.load(open('stacked_gru_sqh','rb'))
# print(list(entropy.keys())) -> val_loss, val_acc, loss, acc

dictionary_keys = list(entropy.keys())
names = [entropy,mse,msle,kl,sqh]





def plots(what = 'bias_variance'):
	n = ['Binary Cross Entropy','Mean Squared Error','MSE Log','KL Divergence','Squared Hinge Loss']
	if str(what) == 'bias_variance':
		# subplots of tr, te losses
		# for each loss method
		pass
	elif str(what) == 'accs':
		# subplots of val_acc vs acc
		# for each loss method
		pass
	elif str(what) == 'compare':
		# plot maximum te accuracy
		# of each loss method
		plt.title("Comparing maximum accuracy achieved on test set")
		x = [i for i in range(0,5)]
		accs = [max(i['val_acc'])*100 for i in names]
		#print(n)
		pl.xticks(x,n)
		pl.xticks(range(5),n,rotation = 45)
		pl.plot(x,accs,'r*')
		pl.show()
	else:
		print("Sorry, don't know what\
			you want me to do.\nSo let's \
			look at all the plots")
		plots('bias_variance')
		plots('accs')
		plots('compare')
plots('compare')