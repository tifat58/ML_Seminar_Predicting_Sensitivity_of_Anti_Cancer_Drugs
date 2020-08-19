from __future__ import print_function

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif




#read feature file, transpose and discard first column
#read response file
#feature selection with anova

nfeat = int(sys.argv[1])
print(nfeat)


feat = pd.read_excel('../data/expression_final4.xlsx')
feat = feat.T[1:]


#get column x
resp = pd.read_excel('../data/response_T_final.xlsx')
resp = resp.T
one = resp.iloc[:,1]
one.to_frame()

#drop all NaN

one.dropna(how='any', inplace=True)

resp = one
#match cosmic ids
indexf = list(feat.index)
indexr = list(resp.index)

for i in range(len(indexf)):
    if (indexf[i] not in indexr):
        feat.drop(index=indexf[i], inplace=True)
    if ('.1' in str(indexf[i])) and (int(str(indexf[i][:-2])) in list(feat.index)):
        s = str(indexf[i][:-2])
        feat.drop(index=int(s), inplace=True)

indexf = list(feat.index)

for j in range(len(indexr)):
    if (indexr.count(indexr[j]) > 1) or (indexr[j] not in indexf):
        resp.drop(index=indexr[j], inplace=True)
    if ('.1' in str(indexr[j])) and (int(str(indexr[j][:-2])) in list(resp.index)):
        s = str(indexr[j][:-2])
        resp.drop(index=int(s), inplace=True)

Y = resp.as_matrix()

#select k best
selector = SelectKBest(f_classif, k = nfeat)

X = selector.fit_transform(feat, Y)




#build model

learning_rate = 0.1
training_epochs = 700
batch_size = 20
display_step = 100
logs_path = '/log_neural_network'

# Network Parameters
n_hidden_1 = nfeat  # 1st layer number of features
n_hidden_2 = nfeat  # 2nd layer number of features
n_input = nfeat  # Number of feature
n_classes = 1  # Number of classes to predict

# tf Graph input and output
x = tf.placeholder("float", [None, n_input])
#tf.summary.scalar("input data", x)
y = tf.placeholder("float", [None, None])
#tf.summary.scalar("response data", y)



# Create model
def multilayer_perceptron(x, weights, biases):
    print("x.shape", x.shape)
    print("weights", weights)

    # Hidden layer with RELU activation : changed to sigmoid
    activ1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    tf.summary.histogram('activation1', activ1)
    layer_1 = tf.nn.sigmoid(activ1)

    # Hidden layer with RELU activation :  changed to sigmoid
    activ2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    tf.summary.histogram('activation2', activ2)
    layer_2 = tf.nn.sigmoid(activ2)

    # Output layer with linear activation
    activ3 = tf.matmul(layer_2, weights['out']) + biases['out']
    tf.summary.histogram('activation3', activ3)
    out_layer = tf.nn.sigmoid(activ3)

    return out_layer


# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([1, n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([1, n_classes]))
}

tf.summary.histogram('weight1', weights['w1'])
tf.summary.histogram('weight2', weights['w2'])
tf.summary.histogram('weight out', weights['out'])

tf.summary.histogram('bias1', biases['b1'])
tf.summary.histogram('bias2', biases['b2'])
tf.summary.histogram('bias out', biases['out'])


print("w1", weights['w1'])
print("b1", biases['b1'])
print("w2", weights['w2'])
print("b2", biases['b2'])


# Construct model
pred = multilayer_perceptron(x, weights, biases)
pred_b = pred > 0.5

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar("cost", cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# accuracy measurement
correct_prediction = tf.equal(tf.cast(pred_b, "float"), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
print('correct ', correct_prediction)
print('acc ', accuracy)
tf.summary.scalar("accuracy", accuracy)



# Initializing the variables
init = tf.global_variables_initializer()


print('X ', X)
print('Y ', Y)
#k-fold thing
data_kv = []
for i in range(len(Y)):
	data_kv.append(np.append(X[i], Y[i]))
data_kv = np.array(data_kv)
print('data kv ', data_kv)
#data_kv = np.append(X, Y, axis=1)
print(data_kv.shape, type(data_kv))
kf = KFold(n_splits=10)
#KV_data = kf.split(data_kv)
#print('kv data shape ', KV_data)
# print(kf.get_n_splits(X_k))
# print(data_kv[:,50])
#for k_train, k_test in kf.split(data_kv):
#     print(k_train, k_test)
#     print("K train:", k_train[:,50:].shape)
    #print(data_kv[k_train][:,20:].shape, type(data_kv[k_train]), len(data_kv[k_train]), data_kv[k_train][1,20:], data_kv[1,1])

accuracy_train = dict()
accuracy_test = dict()
k_count = 1




'''
TN = tf.metrics.true_negatives(y, pred_b)
TP = tf.metrics.true_positives(y, pred_b)
FN = tf.metrics.false_negatives(y, pred_b)
FP = tf.metrics.false_positives(y, pred_b)

acc = tf.metrics.accuracy(y, pred_b)
tf.summary.scalar("train tf_accuracy", acc)
prec = tf.metrics.precision(y, pred_b)
tf.summary.scalar("train tf_precision", prec)

'''


#TP = tf.count_nonzero(tf.cast(pred_b, "float32") * y, dtype=tf.float32)
#TN = tf.count_nonzero((tf.cast(pred_b, "float32") - 1) * (y - 1), dtype=tf.float32)
#FP = tf.count_nonzero(tf.cast(pred_b, "float32") * (y - 1), dtype=tf.float32)
#FN = tf.count_nonzero((tf.cast(pred_b, "float32") - 1) * y, dtype=tf.float32)

#sens = TP / (TP + FN)
#tf.summary.scalar("sensitivity", sens)
#spec = TN / (TN + FP)
#tf.summary.scalar("specificity", spec)


# global result
#     result = tf.argmax(pred, 1).eval({x: X_train, y: Y_train})

# Test accuracy
# Test model on test data

correct_prediction = tf.equal(tf.cast(pred_b, "float32"), y)
#     print("correct prediction", correct_prediction.eval({x: test_X, y: test_Y}))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
#print("Test Accuracy:", accuracy.eval({x: test_X, y: test_Y}))
#tf.summary.scalar("test accuracy", accuracy.eval({x: test_X, y: test_Y}))

#ones = [1] * len(y)
#print (ones)
#zeros = [0] *len(y)
#print (zeros)

print(correct_prediction)


# global result
#     result = tf.argmax(pred, 1).eval({x: test_X, y: test_Y})
merged = tf.summary.merge_all()

# Edited by Hasan
# Launch the graph
for k_train, k_test in kf.split(data_kv):
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(data_kv[k_train]) / batch_size)  # batch size is 9, 9 * 93 = 837
            new_y = []
            new_x = []
            for i in data_kv[k_train]:
                new_x.append(i[:-1])
                new_y.append(i[-1])
            #print('newx ', len(new_x))
            #print('newy ', len(new_y))
            X_batches = np.array_split(new_x, total_batch)
            Y_batches = np.array_split(new_y, total_batch)
            #             X_batches = np.array_split(X, total_batch)
            # print('len batchX ', len(X_batches) , len(X_batches[1]))
            # print('len batchy ', len(Y_batches), len(Y_batches[1]))
            #             Y_batches = np.array_split(Y, total_batch)

            # Loop over all batches
            for i in range(total_batch):
                # print('batch x ', Y_batches[i])
                batch_x, batch_y = X_batches[i], Y_batches[i]
               # print(batch_y.shape)
                 #batch_y = np.array(batch_y)
                batch_y.shape = (batch_y.shape[0], 1)
                # Run optimization op (backprop) and cost op (to get loss value)
                #             print("batch_x", batch_x)

                #             print("w1", weights['w1'].shape)
                #             print("b1", biases['b1'].shape)
                #             print("w2", weights['w2'].shape)
                #             print("b2", biases['b2'].shape)
                #             print("out_w", weights['out'].shape)
                #             print("out_b", biases['out'].shape)

                #             print("out", sess.run(weights['out']))
                #             outlayer_pred = pred.eval(feed_dict = {x: batch_x})
                #             print("outlayer_pred", outlayer_pred)
                #             print("max", outlayer_pred.max())
                #             print("min", outlayer_pred.min())

                _, c, summary = sess.run([optimizer, cost, merged], feed_dict={x: batch_x,
                                                                               y: batch_y})

                summary_writer.add_summary(summary, epoch * total_batch + i)

                #             print("cost c is ", c)
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
        #             if epoch % display_step == 0:
        #                 print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Training accuracy
        X_train = new_x
        Y_train = new_y
        Y_train = np.array(Y_train)
        Y_train.shape = (Y_train.shape[0],1)

        # Test model on training data
        #correct_prediction = tf.equal(tf.cast(pred_b, "float"), y)
        #     print("correct prediction", correct_prediction.eval({x: X_train, y: Y_train}))
        # Calculate accuracy
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        print("Train Accuracy:", accuracy.eval({x: X_train, y: Y_train}))
        train_accuracy = accuracy.eval({x: X_train, y: Y_train})
        accuracy_train.update({k_count: train_accuracy})
        tf.summary.scalar('train acc', train_accuracy)
        
        # global result
        #     result = tf.argmax(pred, 1).eval({x: X_train, y: Y_train})

        # Test accuracy
        # Test model on test data

        #correct_prediction = tf.equal(tf.cast(pred_b, "float32"), y)
        print("correct prediction", correct_prediction.eval({x: X_train, y: Y_train}))
        
#sensitivity 
        o = [1] * Y_train.shape[0]
        pos_p = tf.equal(tf.cast(pred_b, 'float32'),tf.cast(o, 'float32'))
        pos_y = tf.equal(y, tf.cast(o, 'float32'))
        print('positive p', pos_p.eval({x: X_train, y:Y_train}))
        print('positive y', pos_y.eval({x:X_train, y:Y_train}))



# Calculate accuracy
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        new_test_y = []
        new_test_x = []
        for i in data_kv[k_test]:
                new_test_x.append(i[:-1])
                new_test_y.append(i[-1])
        new_test_y = np.array(new_test_y)
        new_test_y.shape = (new_test_y.shape[0],1)
        print("Test Accuracy:", accuracy.eval({x: new_test_x, y: new_test_y}))
        test_accuracy = accuracy.eval({x: new_test_x, y: new_test_y})
        accuracy_test.update({k_count: test_accuracy})
        k_count += 1
        # global result
    #     result = tf.argmax(pred, 1).eval({x: test_X, y: test_Y})

