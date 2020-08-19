##########################################################################
# Start: Library imports
from __future__ import print_function
import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import timeit
from collections import defaultdict
# from tkinter import *
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# End: Library imports
##########################################################################i


###########################################################################
# GPU memory allocation
# print("configuring GPU.....\n\n\n\n")
# os.environ["CUDA_VISIBLE_DEVICES"] = '7' # use the GPU with ID = 0
# config1 = tf.ConfigProto()
# config1.allow_soft_placement = True
# config1.gpu_options.allocator_type = 'BFC'
# config1.gpu_options.per_process_gpu_memory_fraction =  0.95 
# config1.gpu_options.allow_growth = True
# print("GPU configuring done!\n\n\n\n")
##########################################################################


##########################################################################
# INSTRUCTIONS FOR RUNNING THIS FILE
# eg.: python filename.py argument1 argument2 argument3
# 1 argument: number of features, not important for mutation data
# 2 argument: Drug index from response file. Note: Index starts from zero and ignore 1st line while calculating index
# 3 argument: 'mut' if features are mutation data
##########################################################################


##########################################################################
# Start: Read hyperparameters and drug detials rom text file
print("reading parameter file...\n\n\n\n") 
open_file = open('parameters.txt', 'r', encoding='utf-8')
lines     = open_file.read().splitlines()

# Reading features and responses data
# resp_file = lines[0].split('=')[1]
# feature_file = lines[1].split('=')[1]

learning_rate   = float(lines[2].split('=')[1])
training_epochs = int(lines[3].split('=')[1])
batch_size      = int(lines[4].split('=')[1])
display_step    = int(lines[5].split('=')[1])

print("reading parameters file done!\n\n\n\n")
# End:Read Hyperparameters and drug deteils from text file.
##########################################################################


##########################################################################
# Start: Data matrix creation
# read feature file, transpose and discard first column
# read response file
# feature selection with anova

# Get number of features: 1st input argument.
nfeat = int(sys.argv[1])

print("reading response file... \n\n\n\n")
# Get response data 
resp = pd.read_excel('../data/response_T_final.xlsx')
# resp = pd.read_excel(resp_file)

resp = resp.T
name = list(resp)[int(sys.argv[2])]
print('Drugname: ',  list(resp)[int(sys.argv[2])])
one = resp.iloc[:,int(sys.argv[2])]
one.to_frame()
print("reading respnse file done!\n\n\n\n")

# read feature data
print("reading features file...\n\n\n\n")
feat = pd.read_excel('../data/expression_final3.xlsx')
# feat = pd.read_excel(feature_file)
# feat = pd.read_excel('../data/mutation_final_selected.xlsx')
# feat = pd.read_excel('../data/methylation.xlsx')
# feat = pd.read_excel('../data/cnv_tim_final.xlsx')
# feat3 = feat3.T[1:]
feat = feat.T[1:]

# feat_result = pd.concat([feat2, feat3])
# drop all NaN
one.dropna(how='any', inplace=True)
resp = one
print("reading features file done!\n\n\n\n")

#match cosmic ids, drop the ones that do not occur in both
print("Design matix creation...\n\n\n\n")
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
X = feat
print("Design matrix creation done!\n\n\n\n")

#select features for mutation data: only genes that aare mutated in more than x samples
def select_mutated(feat):
    x = feat.sum(axis=0)
    ind = list(feat.columns)
    
    for i in range(len(ind)):
        if x[i] < 100:
            feat.drop(columns=ind[i], inplace=True, axis=1)
    print("Features selected: ", len(feat.columns))
    no = len(feat.columns)
    return feat, no

#select features
if len(sys.argv) == 4:
    X, nfeat = select_mutated(X)
if len(sys.argv) == 5:
    print("cnv")
# End: Data Matrix creation
##########################################################################


##########################################################################
# Hyperparameters: 

# learning_rate = 0.1
# training_epochs = 100
# batch_size = 16
# display_step = 100
dropout_prob = 1.0

##########################################################################


##########################################################################
# other initializations 
logs_path = '../log_neural_network_' + sys.argv[2]
#outputfile = "report_" + sys.argv[2] + "_" + str(nfeat) + "_cnv.txt"

# Network Parameters
#n_hidden_1 = 33  # 1st layer number of features
#n_hidden_2 = 66  # 2nd layer number of features
n_hidden_3 = 50
n_hidden_4 = 50
n_input = nfeat  # Number of feature
n_classes = 1  # Number of classes to predict

##########################################################################


##############################################################
# functions for plotting

def graph_plot(subplot_no, x_value, y_value,  legend_value, marker='-*'):
    plt.subplot(subplot_no)
    plt.plot(x_value, y_value, marker, label=legend_value)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.grid()


def graph_legend(subplot_no, x_label_value, y_label_value, title_value):
    plt.subplot(subplot_no)
    plt.xlabel(x_label_value)
    plt.ylabel(y_label_value)
    plt.title(title_value)


##########################################################################

##########################################################################
def multilayer_perceptron(x, weights, biases):
    print("x.shape", x.shape)
    print("weights", weights)

    # Hidden layer with sigmoid activation and drop out regularization
    activ1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    tf.summary.histogram('activation1', activ1)
    layer_1 = tf.nn.sigmoid(activ1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob = dropout_prob)
    
    # Hidden layer with sigmoid activation and dropout regularization 
    activ2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    tf.summary.histogram('activation2', activ2)
    layer_2 = tf.nn.sigmoid(activ2) 
    layer_2 = tf.nn.dropout(layer_2, keep_prob = dropout_prob)
    
    # Hidden layer with sigmoid activation and dropout regiularization
    activ3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    tf.summary.histogram('activation3', activ3)
    layer_3 = tf.nn.sigmoid(activ3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob = dropout_prob)
    
    # Hidden layer with sigmoid activation and dropout regularization 
    activ4 = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
    tf.summary.histogram('activation4', activ4)
    layer_4 = tf.nn.sigmoid(activ4) 
    layer_4 = tf.nn.dropout(layer_4, keep_prob = dropout_prob)
     
    # Output layer with linear activation
    activ5 = tf.matmul(layer_4, weights['out']) + biases['out']
    tf.summary.histogram('activation5', activ5)
    out_layer = tf.nn.sigmoid(activ5)

    return out_layer
##########################################################################


##########################################################################
# Place holder creation
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, None])
##########################################################################


# gridsearch 
for n_hidden_1 in [50]:
    for n_hidden_2 in [50]:
        outlist = list()
        outlist.append('Number of features: ' + str(nfeat))
        outlist.append('Drug number: ' + sys.argv[2])
        outlist.append('Drug name: ' + name)
        outlist.append('Learning rate: ' + str(learning_rate))
        outlist.append('Training epochs: ' + str(training_epochs))
        outlist.append('Batch size: ' + str(batch_size))
        outputfile = "report_" + sys.argv[2] + "_" + sys.argv[1] + "_h1_" + str(n_hidden_1) + "_h2_" + str(n_hidden_2) + "_exp.txt"

        outlist.append('Number of hidden layers: 2')
        outlist.append('Number of neurons in hidden layer1: ' + str(n_hidden_1))
        outlist.append('Number of neurons in hudden layer2: ' + str(n_hidden_2))

        # tf Graph input and output
        #x = tf.placeholder("float", [None, n_input])
        #tf.summary.scalar("input data", x)
        #y = tf.placeholder("float", [None, None])
        #tf.summary.scalar("response data", y)

        # Store layers weight & bias
        weights = {
            'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), 
            'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'w4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
            'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))

        }

        biases = {
            'b1': tf.Variable(tf.random_normal([1, n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([1, n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([1, n_hidden_3])),
            'b4': tf.Variable(tf.random_normal([1, n_hidden_4])),
            'out': tf.Variable(tf.random_normal([1, n_classes]))
        }

        tf.summary.histogram('weight1', weights['w1'])
        tf.summary.histogram('weight2', weights['w2'])
        tf.summary.histogram('weight out', weights['out'])
        # summary not added for layer 3 and 4 parameters
        tf.summary.histogram('bias1', biases['b1'])
        tf.summary.histogram('bias2', biases['b2'])
        tf.summary.histogram('bias out', biases['out'])

        # Construct model
        pred = multilayer_perceptron(x, weights, biases)
        pred_b = pred > 0.5

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = float(learning_rate)).minimize(cost)
        # accuracy measurement
        correct_prediction = tf.equal(tf.cast(pred_b, "float"), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        # print('correct ', correct_prediction)
        # print('acc ', accuracy)
        tf.summary.scalar("accuracy", accuracy)


        # Initializing the variables
        init = tf.global_variables_initializer()


        #print('X ', X)
        #print('Y ', Y)
        #k-fold cross validation
        data_kv = []
        if len(sys.argv) > 2:
            for i in range(len(Y)):
                 data_kv.append(np.append(X.iloc[i], Y[i]))
        #    data_kv.append(np.append(X[X.columns[i]], Y[i]))
        else:
            for i in range(len(Y)):
                data_kv.append(np.append(X[i], Y[i]))
        data_kv = np.array(data_kv)
        #print('data kv ', data_kv)
        #data_kv = np.append(X, Y, axis=1)
        #print(data_kv.shape, type(data_kv))
        kf = KFold(n_splits=10)

        accuracy_train = dict()
        accuracy_test = dict()
        k_count = 1

        #merge all tensorboard summaries
        merged = tf.summary.merge_all()

        # Launch the graph
        for k_train, k_test in kf.split(data_kv):

            # with tf.Session(config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            cost_list = []
            with tf.Session() as sess:
                start = timeit.default_timer()
                sess.run(init)
                # summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
                summary_writer = tf.summary.FileWriter(logs_path + '/' + str(k_count), graph=sess.graph)

                # Training cycle
                for epoch in range(int(training_epochs)):
                    avg_cost = 0
                    total_batch = int(len(data_kv[k_train]) / batch_size)  # batch size is 9, 9 * 93 = 837
                    new_y = []
                    new_x = []
                    for i in data_kv[k_train]:
                        new_x.append(i[:-1])
                        new_y.append(i[-1])

                    # if expressioon data: select k best
                    if len(sys.argv) == 3:
                        selector = SelectKBest(f_classif, k=nfeat)
                        new_x = selector.fit_transform(new_x, new_y)


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
                         #batch_y = np.array(batch_y)
                        batch_y.shape = (batch_y.shape[0], 1)
                        # Run optimization op (backprop) and cost op (to get loss value)
                        #             print("batch_x", batch_x)

                        _, c, summary = sess.run([optimizer, cost, merged], feed_dict={x: batch_x,
                                                                                       y: batch_y})

                        #w_new, b_new = sess.run([weights, biases])
                        #outlist.append("weights: " + str(w_new))
                        summary_writer.add_summary(summary, epoch * total_batch + i)
                        #outlist.append('Cost: ' + str(c) + 'kfold ' + str(k_count))

                        #             print("cost c is ", c)
                        # Compute average loss

                        avg_cost += c / total_batch

                    cost_list.append(avg_cost)
                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                print("Optimization Finished!")

                #plotting
                # plt.plot(range(1,len(cost_list)+1), cost_list, '*-')
                legend_value = 'Fold_' + str(k_count)
                graph_plot(211, range(1, len(cost_list)+1), cost_list, legend_value)

                # Training accuracy
                X_train = new_x
                Y_train = new_y
                Y_train = np.array(Y_train)
                Y_train.shape = (Y_train.shape[0],1)

                # Test model on training data
                # Calculate accuracy
                print("Train Accuracy:", accuracy.eval({x: X_train, y: Y_train}))
                train_accuracy = accuracy.eval({x: X_train, y: Y_train})
                accuracy_train.update({k_count: train_accuracy})
                outlist.append('\n' + 'Fold: ' + str(k_count))
                outlist.append('Train Accuracy: ' + str(accuracy.eval({x: X_train, y: Y_train})))
                #tf.summary.scalar('train acc', train_accuracy)


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

                stop = timeit.default_timer() # Training time end
                print("Time:", stop - start, "seconds" '\n') # training time display



               # Test model on test data
               # Calculate test accuracy
                new_test_y = []
                new_test_x = []
                for i in data_kv[k_test]:
                        new_test_x.append(i[:-1])
                        new_test_y.append(i[-1])

                #if exrpression data, select k best of test set
                if len(sys.argv) == 3:
                    new_test_x = selector.fit_transform(new_test_x, new_test_y)

                new_test_y = np.array(new_test_y)
                new_test_y.shape = (new_test_y.shape[0],1)
                print("Test Accuracy:", accuracy.eval({x: new_test_x, y: new_test_y}))
                test_accuracy = accuracy.eval({x: new_test_x, y: new_test_y})
                accuracy_test.update({k_count: test_accuracy})
                outlist.append('Test Accuracy: ' + str(accuracy.eval({x: new_test_x, y: new_test_y})))
                k_count += 1

                #add weights and biases
                weights2, biases2 = sess.run([weights, biases])

                for w in weights2:
                    outlist.append(str(w) + ': ' + str(weights2[w].tolist()))

                for b in biases2:
                    outlist.append(str(b) + ': ' + str(biases2[b].tolist()))


        graph_plot(212, range(1, len(accuracy_train.keys())+1), accuracy_train.values(), 'Train Accuracy')
        graph_plot(212, range(1, len(accuracy_test.keys())+1), accuracy_test.values(), 'Test Accuracy')
        graph_legend(212, 'Folds', 'Accuracy', 'Train and Test Accuracy vs K-Folds')

        graph_legend(211, 'No. of Epoch', 'Cost value', 'Cost value in each fold')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        figure_folder = "../Figure/Figure_" + sys.argv[2] + "_" + sys.argv[1] + "_h1_" + str(n_hidden_1) + "_h2_" + str(n_hidden_2)
        fig_name = str(k_count) + '_fold.png'
        file_loc = figure_folder + '_' + fig_name
        plt.savefig(file_loc, format='png', dpi=1000, bbox_inches='tight')
        plt.close()
            # plt.show()

        #write report file
        with open(outputfile, 'w') as o:
            for element in outlist:
                o.write(element)
                o.write("\n")
