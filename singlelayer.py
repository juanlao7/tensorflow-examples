#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N


#read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
#print ( N.shape(data[0][0])[0] )
#print ( N.shape(data[0][1])[0] )

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  

#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0

import matplotlib.pyplot as plt

def plotResults(results):
    plt.gca().set_color_cycle(None)
    
    for result in results:
        plt.plot(result[1])
    
    plt.legend([result[0] for result in results], loc='upper right')
    
    plt.ylabel('Cross-entropy')
    plt.xlabel('Epoch')
    
    #plt.ylim(ymin=-1, ymax=25)
    
    plt.show()

experiment = 2

if experiment == 1:
    # Different steps
    
    optimizers = [
        ['Step 0.9', tf.train.GradientDescentOptimizer(0.9)],
        ['Step 0.5', tf.train.GradientDescentOptimizer(0.5)],
        ['Step 0.05', tf.train.GradientDescentOptimizer(0.05)],
        ['Step 0.005', tf.train.GradientDescentOptimizer(0.005)],
        ['Step 0.0005', tf.train.GradientDescentOptimizer(0.0005)],
    ]
else:
    # Different optimizers
    
    optimizers = [
        ['Gradient Descent', tf.train.GradientDescentOptimizer(0.01)],
        ['AdadeltaOptimizer', tf.train.AdadeltaOptimizer(0.01)],
        ['AdagradOptimizer', tf.train.AdagradOptimizer(0.01)],
        ['AdamOptimizer', tf.train.AdamOptimizer(0.01)],
        ['RMSPropOptimizer', tf.train.RMSPropOptimizer(0.01)]
    ]

results = []

for config in optimizers:
    optimizerName = config[0]
    
    print('####', optimizerName, '####')
    #set up the computation. Definition of the variables.
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    
    train_step = config[1].minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    #TRAINING PHASE
    print("TRAINING")
    
    result = []
    
    for i in range(500):
      batch_xs = data[0][0][100*i:100*i+100]
      batch_ys = real_output[100*i:100*i+100]
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      curr_cross_entropy = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}) 
      
      result.append(curr_cross_entropy)
    
    
    #CHECKING THE ERROR
    print("ERROR CHECK")
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))
    
    results.append([optimizerName, result])

print('#### RESULTS ####')

for result in results:
    print(result[0], ':', result[1][-1])

plotResults(results)
