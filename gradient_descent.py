#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt

def plotResults(results):
    plt.gca().set_color_cycle(None)
    
    for result in results:
        plt.plot(result[1])
    
    plt.legend([result[0] for result in results], loc='upper right')
    
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.ylim(ymin=-1, ymax=25)
    
    plt.show()

experiment = 2

if experiment == 1:
    # Different steps
    
    optimizers = [
        ['Step 0.025', tf.train.GradientDescentOptimizer(0.025)],
        ['Step 0.01', tf.train.GradientDescentOptimizer(0.01)],
        ['Step 0.001', tf.train.GradientDescentOptimizer(0.001)],
        ['Step 0.0001', tf.train.GradientDescentOptimizer(0.0001)],
        ['Step 0.00001', tf.train.GradientDescentOptimizer(0.0001)],
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
    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
    # optimizer
    optimizer = config[1]
    train = optimizer.minimize(loss)
    
    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    
    result = []
    
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        
        if i % 100 == 0:
            print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
        
        result.append(curr_loss)
    
    results.append([optimizerName, result])

print('#### RESULTS ####')

for result in results:
    print(result[0], ':', result[1][-1])

plotResults(results)

    
