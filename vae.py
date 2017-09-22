import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

n_pixels = 28*28

x=tf.placeholder(tf.float32,[None,n_pixels])

def weights_variables(shape,name):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial,name=name)

def bias_variable(shape,name):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial,name=name)

def FC_layer(X,W,b):
	return tf.matmul(X,W)+b

latent_dim = 20
h_dim = 500

#layer 1

W_enc = weights_variables([n_pixels,h_dim],'W_enc')
b_enc = bias_variable([h_dim],'b_enc')

#tanh

h_enc = tf.nn.tanh(FC_layer(x,W_enc,b_enc))

#layer 2

W_mu = weights_variables([h_dim,latent_dim],'W_mu')
b_mu = bias_variable([latent_dim],'b_mu')
mu = FC_layer(h_enc,W_mu,b_mu) #mean

#standard Deviation

W_logstd = weights_variables([h_dim,latent_dim],'W_logstd')
b_logstd = bias_variable([latent_dim],'b_logstd')
logstd = FC_layer(h_enc,W_logstd,b_logstd) #std

#RANDOM

noise = tf.random_normal([1,latent_dim])

#Output of the encoder

z = mu+tf.multiply(noise,tf.exp(.5*logstd))

#decoder

W_dec = weights_variables([latent_dim,h_dim],'W_dec')
b_dec = weights_variables([h_dim],'b_dec')

#pass in z here

h_dec = tf.nn.tanh(FC_layer(z,W_dec,b_dec))

#layer2 , using original n pixels here since thats the dimensionality

W_reconstruct = weights_variables([h_dim,n_pixels],'W_reconstruct')
b_reconstruct = bias_variable([n_pixels],'b_reconstruct')

#784 bernoulli parameters output

reconstruction = tf.nn.sigmoid(FC_layer(h_dec,W_reconstruct,b_reconstruct))

# variational lower bound

# add epsilon to log to prevent numerical overflow
#Information is lost because it goes from a smaller to a larger dimensionality. 
#How much information is lost? We measure this using the reconstruction log-likelihood 
#This measure tells us how effectively the decoder has learned to reconstruct
#an input image x given its latent representation z.
log_likelihood = tf.reduce_sum(x*tf.log(reconstruction + 1e-9)+(1 - x)*tf.log(1 - reconstruction + 1e-9), reduction_indices=1)
#KL Divergence
#If the encoder outputs representations z that are different 
#than those from a standard normal distribution, it will receive 
#a penalty in the loss. This regularizer term means 
#keep the representations z of each digit sufficiently diverse. 
#If we didnt include the regularizer, the encoder could learn to cheat
#and give each datapoint a representation in a different region of Euclidean space. 
KL_term = -.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu,2) - tf.exp(2*logstd), reduction_indices=1)

# This allows us to use stochastic gradient descent with respect to the variational parameters
variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)

#init all variables and start the session!
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
## Add ops to save and restore all the variables.
saver = tf.train.Saver()

import time #lets clock training time..

num_iterations = 1000
recording_interval = 10
#store value for these 3 terms so we can plot them later
variational_lower_bound_array = []
log_likelihood_array = []
KL_term_array = []
iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]
for i in range(num_iterations):
    # np.round to make MNIST binary
    #get first batch (200 digits)
    x_batch = np.round(mnist.train.next_batch(200)[0])
    #run our optimizer on our data
    sess.run(optimizer, feed_dict={x: x_batch})
    if (i%recording_interval == 0):
        #every 1K iterations record these values
        vlb_eval = variational_lower_bound.eval(feed_dict={x: x_batch})
        print "Iteration: {}, Loss: {}".format(i, vlb_eval)
        variational_lower_bound_array.append(vlb_eval)
        log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={x: x_batch})))
        KL_term_array.append(np.mean(KL_term.eval(feed_dict={x: x_batch})))
    saver.save(sess,'my_test_model',global_step=1000)