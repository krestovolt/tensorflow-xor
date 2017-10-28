import tensorflow as tf
import numpy as np

def make_layer(input_data, input_size, output_size, layer_name,activation_fn = None):
    with tf.name_scope("Layer"+layer_name):
        with tf.name_scope("Weights"+layer_name):
            W = tf.Variable(tf.random_normal([input_size, output_size]), name="Weights")                
            tf.summary.histogram("Weight",W)            
        with tf.name_scope("Biases"+layer_name):
            bias = tf.Variable(tf.zeros([1, output_size]) + 0.1, name="Biases")        
            tf.summary.histogram("Bias",bias)
        with tf.name_scope("Output"+layer_name):
            W_x_plus_bias = tf.matmul(input_data, W, name="W_kali_X_tambah_bias") + bias                
            output_data = None
            if activation_fn is None:
                output_data = W_x_plus_bias#linear
            else:
                output_data = activation_fn(W_x_plus_bias)    
        tf.summary.histogram("Output",output_data)
        return output_data

data = np.load("output.out.npy")

x1 = data[:,[0,]]
x2 = data[:,[1,]]
x = np.column_stack((x1,x2))
y = data[:,[2,]]

print(x)

#test validity
for i in range(100):
    assert(data[i][0] == x[i][0])    
    assert(data[i][1] == x[i][1])
    assert(data[i][2] == y[i])

#define placeholder for input output
#2 feature input
xs = tf.placeholder(tf.float32, [None, 2], name="input_placeholder")
#1 output
ys = tf.placeholder(tf.float32, [None, 1], name="output_placeholder")

#hidden layer
l1 = make_layer(xs, 2, 6, "_hidden",activation_fn = tf.nn.relu)
#output layer
y_out = make_layer(l1, 6, 1, "_output",activation_fn = None)
#hitung loss(atau error)
#reduced mean squared error (y - y')^2 * 1/2
with tf.name_scope("Error"):
    error = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y_out), reduction_indices=[1]))
    tf.summary.histogram("Error",error)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(error)
#initialize all var
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:    
    sess.run(init)
    #define writer
    writer = tf.summary.FileWriter("logs/", sess.graph)

    test_data = np.array([
        [1,0],
        [0,1],
        [0,0],
        [1,1]
    ],dtype = np.float32)

    test_data_out = np.array([
        [1],
        [1],
        [0],
        [0]
    ],dtype = np.float32)

    for step in range(500):        
        if step % 10 == 0:
            summary, acc = sess.run([merged, error], feed_dict ={xs: test_data, ys: test_data_out})
            writer.add_summary(summary,step)
            print('error at step ',step,'=>',sess.run(error, feed_dict = {xs: x, ys: y}))
        else:
            summary, acc = sess.run([merged, train_step], feed_dict = {xs: x, ys: y})
            writer.add_summary(summary,step)        
        


    feed_dict = {xs: test_data}
    prediction = sess.run(y_out, feed_dict)
    prediction = list(map(lambda x: 1 if x>0.5 else 0, prediction))    
    for i,input in enumerate(test_data):
        print(input, " => ",prediction[i])

