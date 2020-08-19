from __future__ import print_function

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tkinter import *
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# read feature file, transpose and discard first column
# read response file
# feature selection with anova


# 1 argument: features, not important for mutation data
# 2 argument: which drug
# 3 argument: 'mut' if features are mutation data

nfeat = int(sys.argv[1])
# print(nfeat)

# get drug
resp = pd.read_excel('../data/response_T_final.xlsx')
# print(resp)
resp = resp.T
name = list(resp)[int(sys.argv[2])]
print('Drugname: ', list(resp)[int(sys.argv[2])])
one = resp.iloc[:, int(sys.argv[2])]
one.to_frame()

# read feature data
feat = pd.read_excel('../data/expression_final3.xlsx')
mut = pd.read_excel('../data/mutation_final_selected.xlsx')
# feat = pd.read_excel('../data/methylation.xlsx')
cnv = pd.read_excel('../data/cnv_tim_final.xlsx')
# feat3 = feat3.T[1:]
feat = feat.T[1:]
mut = mut.T
cnv = cnv.T
print('finished reading')

# drop all NaN
one.dropna(how='any', inplace=True)
resp = one

# match cosmic ids, drop the ones that do not occur in both
indexf = list(feat.index)
#print(indexf)
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

if len(sys.argv) == 3:
    selector = SelectKBest(f_classif, k=nfeat)
    feat = selector.fit_transform(feat, resp)
    print(feat)
    feat = pd.DataFrame(feat)#, index=ind, columns=cols)

feat_result = pd.concat([feat, mut, cnv], join='inner', axis=1)
feat_result = feat_result.dropna(how='any')

X = feat_result
nfeat = len(X.columns)
print("after dropout: " + str(nfeat))

# build model
learning_rate = 0.1
training_epochs = 100
batch_size = 16
display_step = 100
logs_path = '../log_neural_network_' + sys.argv[2]
# outputfile = "report_" + sys.argv[2] + "_" + str(nfeat) + "_cnv.txt"


# Network Parameters
# n_hidden_1 = 33  # 1st layer number of features
# n_hidden_2 = 66  # 2nd layer number of features
n_input = nfeat  # Number of feature
n_classes = 1  # Number of classes to predict


# Create model
def multilayer_perceptron(x, weights, biases, n_hidden):
    print("x.shape", x.shape)
    print("weights", weights)

    # Hidden layer with RELU activation : changed to sigmoid
    previous = x

    for i in range(1, n_hidden):
        activ1 = tf.add(tf.matmul(previous, weights['w' + str(i)]), biases['b' + str(i)])
        # tf.summary.histogram('activation1', activ1)
        layer_1 = tf.nn.sigmoid(activ1)
        previous = layer_1

    # Hidden layer with RELU activation :  changed to sigmoid
    # activ2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    # tf.summary.histogram('activation2', activ2)
    # layer_2 = tf.nn.sigmoid(activ2)

    # Output layer with linear activation
    activ3 = tf.matmul(previous, weights['out']) + biases['out']
    tf.summary.histogram('activation3', activ3)
    out_layer = tf.nn.sigmoid(activ3)

    return out_layer


def shuffle_train_data(X_train, Y_train):
    """called after each epoch"""
    perm = np.random.permutation(len(Y_train))
    Xtr_shuf = X_train[perm]
    Ytr_shuf = Y_train[perm]

    return Xtr_shuf, Ytr_shuf


# for overview file
summ = list()
summfile = "summary_" + sys.argv[2] + "_" + sys.argv[1] + "_layers3" + "_combined.txt"

x = tf.placeholder("float", [None, None])
y = tf.placeholder("float", [None, None])


# gridsearch
for n_layers in [2, 3]:#, 4, 5, 6, 7]:
    outlist = list()
    outlist.append('Number of features: ' + str(nfeat))
    outlist.append('Drug number: ' + sys.argv[2])
    outlist.append('Drug name: ' + name)
    outlist.append('Learning rate: ' + str(learning_rate))
    outlist.append('Training epochs: ' + str(training_epochs))
    outlist.append('Batch size: ' + str(batch_size))
    outputfile = "report_" + sys.argv[2] + "_" + sys.argv[1] + "_layers_" + str(n_layers) + "_combined.txt"
    outlist.append('Number hidden layers: ' + str(n_layers))

    # equal number of hidden neurons
    n_hidden = 100

    # first and last weight is set before
    weights = {'w1': tf.Variable(tf.random_normal([n_input, n_hidden]))}

    for i in range(2, n_layers + 1):
        weights['w' + str(i)] = tf.Variable(tf.random_normal([n_hidden, n_hidden]))

    weights['out'] = tf.Variable(tf.random_normal([n_hidden, n_classes]))

    # only last bias is set
    biases = {}

    for i in range(1, n_layers + 1):
        biases['b' + str(i)] = tf.Variable(tf.random_normal([1, n_hidden]))

    biases['out'] = tf.Variable(tf.random_normal([1, n_classes]))

    # needs to be adapted when different neuron counts
    outlist.append('Number hidden neurons: ' + str(n_hidden))

    summ.append('Number hidden layers: ' + str(n_layers))
    summ.append('Number hidden neurons: ' + str(n_hidden))

    # tf.summary.histogram('weight1', weights['w1'])
    # tf.summary.histogram('weight2', weights['w2'])
    # tf.summary.histogram('weight out', weights['out'])

    # tf.summary.histogram('bias1', biases['b1'])
    # tf.summary.histogram('bias2', biases['b2'])
    # tf.summary.histogram('bias out', biases['out'])

    # print("w1", weights['w1'])
    # print("b1", biases['b1'])
    # print("w2", weights['w2'])
    # print("b2", biases['b2'])

    # Construct model
    pred = multilayer_perceptron(x, weights, biases, n_layers)
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

    # print('X ', X)
    # print('Y ', Y)
    # k-fold cross validation
    data_kv = []
    if len(sys.argv) > 2:
        for i in range(len(Y)):
            data_kv.append(np.append(X.iloc[i], Y[i]))
    #    data_kv.append(np.append(X[X.columns[i]], Y[i]))
    else:
        for i in range(len(Y)):
            data_kv.append(np.append(X[i], Y[i]))
    data_kv = np.array(data_kv)
    # print('data kv ', data_kv)
    # data_kv = np.append(X, Y, axis=1)
    # print(data_kv.shape, type(data_kv))
    kf = KFold(n_splits=10)

    accuracy_train = dict()
    accuracy_test = dict()
    k_count = 1

    # merge all tensorboard summaries
    merged = tf.summary.merge_all()

    # Launch the graph
    for k_train, k_test in kf.split(data_kv):

        with tf.Session() as sess:
            # start = timeit.default_timer()
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
                # print('newx ', len(new_x))
                # print('newy ', len(new_y))

                new_x, new_y = shuffle_train_data(new_x, new_y)

                # if expressioon data: select k best
                #if len(sys.argv) == 3:
                #    selector = SelectKBest(f_classif, k=nfeat)
                #    new_x = selector.fit_transform(new_x, new_y)

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
                    # print(batch_x.shape)
                    # print(batch_y.shape)
                    # batch_y = np.array(batch_y)
                    batch_y.shape = (batch_y.shape[0], 1)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    #             print("batch_x", batch_x)

                    _, c, summary = sess.run([optimizer, cost, merged], feed_dict={x: batch_x,
                                                                                   y: batch_y})

                    # w_new, b_new = sess.run([weights, biases])
                    # outlist.append("weights: " + str(w_new))
                    summary_writer.add_summary(summary, epoch * total_batch + i)
                    # outlist.append('Cost: ' + str(c) + 'kfold ' + str(k_count))

                    #             print("cost c is ", c)
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
            #             if epoch % display_step == 0:
            #                 print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # print("Optimization Finished!")

            # Training accuracy
            X_train = new_x
            Y_train = new_y
            Y_train = np.array(Y_train)
            Y_train.shape = (Y_train.shape[0], 1)

            # Test model on training data
            # Calculate accuracy
            print("Train Accuracy:", accuracy.eval({x: X_train, y: Y_train}))
            train_accuracy = accuracy.eval({x: X_train, y: Y_train})
            accuracy_train.update({k_count: train_accuracy})
            outlist.append('\n' + 'Fold: ' + str(k_count))
            outlist.append('Train Accuracy: ' + str(accuracy.eval({x: X_train, y: Y_train})))

            summ.append('\n' + 'Fold: ' + str(k_count))
            summ.append('Train Accuracy: ' + str(accuracy.eval({x: X_train, y: Y_train})))
            # tf.summary.scalar('train acc', train_accuracy)

            # Sensitivity and Specifity
            pred_labels = pred_b.eval({x: X_train, y: Y_train})
            pred_labels = pred_labels.astype(int)
            true_labels = Y_train
            TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
            TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
            FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
            FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

            sensitivity = TP / (TP + FN)
            specifity = TN / (TN + FP)

            # Display sensitivity and specificity
            print('sensitivity:', sensitivity)
            print('specifity', specifity)
            outlist.append('Sensitivity: ' + str(sensitivity))
            outlist.append('Specificity: ' + str(specifity))

            summ.append('Sensitivity: ' + str(sensitivity))
            summ.append('Specificity: ' + str(specifity))

            # stop = timeit.default_timer() # Training time end
            # print("Time:", stop - start, "seconds" '\n') # training time display

            # Test model on test data
            # Calculate test accuracy
            new_test_y = []
            new_test_x = []
            for i in data_kv[k_test]:
                new_test_x.append(i[:-1])
                new_test_y.append(i[-1])

            # if exrpression data, select k best of test set
            if len(sys.argv) == 3:
                new_test_x = selector.fit_transform(new_test_x, new_test_y)

            new_test_y = np.array(new_test_y)
            new_test_y.shape = (new_test_y.shape[0], 1)
            print("Test Accuracy:", accuracy.eval({x: new_test_x, y: new_test_y}))
            test_accuracy = accuracy.eval({x: new_test_x, y: new_test_y})
            accuracy_test.update({k_count: test_accuracy})
            outlist.append('Test Accuracy: ' + str(accuracy.eval({x: new_test_x, y: new_test_y})))
            summ.append('Test Accuracy: ' + str(accuracy.eval({x: new_test_x, y: new_test_y})))
            k_count += 1

            # add weights and biases
            weights2, biases2 = sess.run([weights, biases])

            for w in weights2:
                outlist.append(str(w) + ': ' + str(weights2[w].tolist()))

            for b in biases2:
                outlist.append(str(b) + ': ' + str(biases2[b].tolist()))

    # write report file
    with open(outputfile, 'w') as o:
        for element in outlist:
            o.write(element)
            o.write("\n")

    # write summary file
    with open(summfile, 'w') as o:
        for element in summ:
            o.write(element)
            o.write("\n")
