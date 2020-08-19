from __future__ import print_function

import pickle
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from collections import defaultdict
# import matplotlib.pyplot as plt
# from tkinter import *
# from __builtin__ import file
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import f1_score
import matplotlib
import timeit
matplotlib.use('agg')
import matplotlib.pyplot as plt
from itertools import permutations, product


# GPU usage
gpu = sys.argv[3]
type = 'exp'
# for ix in [1]:
with tf.device('/device:GPU:' + str(gpu)):
    # 1 argument: number of features
    # 2 argument: which drug
    # 3 argument: GPU
    # 4. argument: lower number of total layers
    # 5. argument: upper number of total layers
    # 6. argument: how many parts of combination list (for whole list : 1 0)
    # 7. argument: which part starting with 0 (all parts need to be included to have the whole list)

    def shuffle_train_data(X_train, Y_train):
        """called after each epoch"""
        perm = np.random.permutation(len(Y_train))
        Xtr_shuf = X_train[perm]
        Ytr_shuf = Y_train[perm]

        return Xtr_shuf, Ytr_shuf


    ######################################################################
    # functions for plotting

    def graph_plot(subplot_no, x_value, y_value, legend_value, marker='-*'):
        plt.subplot(subplot_no)
        plt.plot(x_value, y_value, marker, label=legend_value)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0))

        plt.grid(True)


    def graph_legend(subplot_no, x_label_value, y_label_value, title_value):
        plt.subplot(subplot_no)
        plt.xlabel(x_label_value)
        plt.ylabel(y_label_value)
        plt.title(title_value)
        # plt.grid()


    ##########################################################################

    def statistics(values):
        """computes mean and std for different sets"""

        stats = []

        # train acc, test acc, senseitivity, specificity
        # first mean, then std

        for val in values:
            stats.append(np.mean(np.array(val)))
            stats.append(np.std(np.array(val)))

        return stats


    # Create model with x as featues, weights, biases and total number of layers
    def multilayer_perceptron(x, weights, biases, n_hidden, drop_prob=1.0):

        # Hidden layer with RELU activation : changed to sigmoid
        previous = x

        for i in range(1, n_hidden - 1):
            activ1 = tf.add(tf.matmul(previous, weights['w' + str(i)]), biases['b' + str(i)])
            activ1 = tf.nn.dropout(activ1, keep_prob=drop_prob)
            # tf.summary.histogram('activation1', activ1)
            layer_1 = tf.nn.sigmoid(activ1)
            previous = layer_1
            if i == n_hidden - 2:
                last = layer_1

        # Output layer with linear activation
        activ3 = tf.matmul(last, weights['out']) + biases['out']
        # tf.summary.histogram('activation3', activ3)
        out_layer = tf.nn.sigmoid(activ3)

        return out_layer


    # number of features to select
    nfeat = int(sys.argv[1])


    ##############
    # best f score

    bestf = 0
    best_model_name = ''
    best_wf = 0
    best_wf_model_name = ''

    ####################
    # read drug response file
    # transpose
    # take nth drug which is specified in argv 2
    resp = pd.read_excel('../data/response_T_final.xlsx')
    resp = resp.T
    name = list(resp)[int(sys.argv[2])]
    print('Drugname: ', list(resp)[int(sys.argv[2])])
    one = resp.iloc[:, int(sys.argv[2])]
    one.to_frame()
    one.dropna(how='any', inplace=True)
    resp = one

    # read feature data file
    # transposed and d line (gene names) discarded
    #feat = pd.read_excel('../data/expression_final3.xlsx')
    #feat = feat.T[1:]
    #with open('exp.file', 'wb') as f:
    #    pickle.dump(feat, f, pickle.HIGHEST_PROTOCOL)
    feat = None
    with open('exp.file', 'rb') as f:
        feat = pickle.load(f)
    # feat = pd.read_excel('../data/mutation_final_selected.xlsx')
    # feat = pd.read_excel('../data/methylation.xlsx')
    # feat = pd.read_excel('../data/cnv_tim_final.xlsx')
    # feat3 = feat3.T[1:]
    # feat = feat.T[1:]
    print('finished reading', len(feat.columns))

    #################
    # match cosmic ids, drop the ones that do not occur in both
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
    binarizer = preprocessing.Binarizer(threshold=0)
    Y = binarizer.transform(np.array(Y).reshape(1, -1))

    Y = Y[0]
    X = np.array(feat)
    # nfeat = len(feat.columns)
    # print("after dropout: " + str(nfeat))

    ###############
    # specify model parameters
    learning_rate = 0.1 #Modified
    training_epochs = 400 #Modified
    batch_size = 30
    display_step = 100
    drop_val = 0.5
    # logs_path = '../log_neural_network_' + sys.argv[2]
    # outputfile = "report_" + sys.argv[2] + "_" + str(nfeat) + "_cnv.txt"

    # Network Parameters
    n_input = nfeat  # Number of feature
    n_classes = 1  # Number of classes to predict

    #################
    # session configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allocator_type = "BFC"
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True

    # for overview file
    summ = list()
    stats = []
    summfile = "../reports2/summary_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep400_ba30_exp_lr_" + str(learning_rate) + '.txt'
    #summfile_figure = "summary_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep400_ba30_exp_lr_" + str(learning_rate) + '.txt'

    new_directory = "../reports2/Drug_" + sys.argv[2]
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)

    new_report_file = 'report_drug_' + sys.argv[2] + '_features_' + sys.argv[1] + '_hlayers_' + sys.argv[4] + '_' + sys.argv[5] + '_part' + sys.argv[7] + '.txt'
    new_summ_file = 'summary_drug_' + sys.argv[2] + '_features_' + sys.argv[1] + '_hlayers_' + sys.argv[4] + '_' + sys.argv[5] + '_part' + sys.argv[7] + '.txt'
    best_model_file = 'Best_model_drug_' + sys.argv[2] + '_features_' + sys.argv[1] + '_hlayers_' + sys.argv[4] + '_' + sys.argv[5] + '_part' + sys.argv[7] + '.txt'

    new_report_path = os.path.join(new_directory, new_report_file)

    file_counter = 1
    while os.path.exists(new_report_path):
        temp_file = new_report_file[:-4] + '_' + str(file_counter) + '.txt'
        new_report_path = os.path.join(new_directory, temp_file)
        file_counter += 1

    print('Report file location:    ' + new_report_path)

    best_model_path = os.path.join(new_directory, best_model_file)
    file_counter = 1

    while os.path.exists(best_model_path):
        temp_file = best_model_file[:-4] + '_' + str(file_counter) + '.txt'
        best_model_path = os.path.join(new_directory, temp_file)
        file_counter += 1

    print('Best Model file location:    ' + best_model_path)

    new_report_dict = dict()
    new_summ_dict = dict()

    new_train_acc_dict = dict()
    new_test_acc_dict = dict()
    new_train_sens_dict = dict()
    new_test_sens_dict = dict()
    new_train_spec_dict = dict()
    new_test_spec_dict = dict()
    new_f_score_dict = dict()
    new_f_w_score_dict = dict()
    new_train_acc_std_dict = dict()
    new_test_acc_std_dict = dict()
    new_train_sens_std_dict = dict()
    new_test_sens_std_dict = dict()
    new_train_spec_std_dict = dict()
    new_test_spec_std_dict = dict()

    new_weights_dict = dict()
    new_biases_dict = dict()

    # Tensorflow placeholders for features and repsonse
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, None])
    drop_prob = tf.placeholder('float', None)

    # create saver
    # saver = tf.train.Saver()
    cv_time_start = timeit.default_timer()

    #list_learning_range = [0.100, 0.010, 0.001]
    list_learning_range = [0.010]
    list_epoch = [200]
    list_dropout = [1.0]
    list_nr_percentage = [0.8]

    parts = int(sys.argv[6])
    part_number = int(sys.argv[7])

    list_nr_hidden = []
    no = nfeat
    while no > 10:
        list_nr_hidden.append(no)
        no = no - 20

    hidden = []


    #create list of all combinations beforehand, put in one big list and iterate over that list

    # loop over number of hidden layers
    for n_layers in range(int(sys.argv[4]), int(sys.argv[5]) + 1):

        # create all combinations of 20 stepsize
        # drop not valid ones with specific conditions
        allcomb = list(product(list_nr_hidden, repeat=n_layers))
        allcomb = list(set(allcomb))

        validcomb = []

        for comb in allcomb:
            valid = True

            #if comb[0] <= 0.5 * nfeat:
            #    valid = False
            #if n_layers > 3 and comb[2] <= 0.5 * nfeat:
            #    valid = False
            #if n_layers > 5 and comb[3] <= 0.5 * nfeat:
            #    valid = False

            if valid:
                for i in range(1, len(comb)):
                    if comb[i - 1] < comb[i]:
                        valid = False
                if not valid:
                    continue
            else:
                continue

            validcomb.append(comb)

        hidden_list = []

        for combi in validcomb:
            hidden = []
            hidden.append(nfeat)
            hidden = hidden + list(combi)
            hidden.append(1)

            hidden_list.append(hidden)

        number_combinations = len(hidden_list)
        parts_list = []
        divide = int(math.ceil(float(number_combinations) / float(parts)))
        for i in range(parts-1):
            start = i * divide
            stop = (i+1) * divide
            parts_list.append((start, stop))

        start = (parts-1) * divide
        stop = number_combinations
        parts_list.append((start, stop))

        for count_lr in list_learning_range:
            learning_rate = count_lr

            for count_epoch in list_epoch:
                training_epochs = count_epoch

                for count_drop in list_dropout:
                    drop_val = count_drop

                    for hidden_neurons in hidden_list[parts_list[part_number][0]:parts_list[part_number][1]]:
                        # make for loops for each layer, have list inputfeatures - 20 steps until 20
                        # if previous one smaller (or equal) than current one, go on to the next one
                        print(hidden_neurons)

                        # initialization of report lists
                        outlist = list()
                        trainaccs = []
                        testaccs = []
                        trainsens = []
                        trainspec = []
                        testsens = []
                        testspec = []

                        model_name = 'feat_' + sys.argv[1] + '__' + str(n_layers) + '_layer_' + str(hidden_neurons) + '_epoch_' + str(count_epoch) + '_drop_' + str(count_drop) + '_lr_' + str(count_lr)

                        ################################################################
                        # creating folders to store plot for each of the grid search
                        if not os.path.exists('../Figure2'):
                            os.mkdir('../Figure2')

                        dir_path = "../Figure2/" + "drug_" + str(sys.argv[2]) + "_features_" + str(sys.argv[1]) + "_epoch_" + str(
                            training_epochs) + "_nLayers_" + str(n_layers) + "_" + str(sys.argv[6]) + 'lr_' + str(learning_rate) + "_Momemtum"

                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)
                            print('Figure directory created')

                        ######################################################################
                        # writing to report files
                        outlist.append('Number of features: ' + str(nfeat))
                        outlist.append('Drug number: ' + sys.argv[2])
                        outlist.append('Drug name: ' + name)
                        outlist.append('Learning rate: ' + str(learning_rate))
                        outlist.append('Training epochs: ' + str(training_epochs))
                        outlist.append('Batch size: ' + str(batch_size))
                        outputfile = "../reports2/report_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep500_ba30_" + str(
                            n_layers) + "_exp_lr_" + str(learning_rate) + ".txt"
                        outputfile_figure = "report_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep500_ba30_" + str(
                            n_layers) + "_exp_lr_" + str(learning_rate) + ".txt"
                        # summ.append('Number of layers: ' + str(n_layers))
                        outlist.append('Number of layers: ' + str(n_layers))
                        summ.append('Number of layers: ' + str(n_layers))

                        # weight and bias setting: last one is set afterwards
                        weights = {}
                        for i in range(1, n_layers+1):
                            # print(np.floor(hidden_neurons[i-1]))
                            weights['w' + str(i)] = tf.Variable(
                                tf.random_normal([hidden_neurons[i - 1], hidden_neurons[i]]))

                        weights['out'] = tf.Variable(tf.random_normal([hidden_neurons[-2], n_classes]))

                        biases = {}
                        for i in range(1, n_layers + 1):
                            biases['b' + str(i)] = tf.Variable(tf.random_normal([1, hidden_neurons[i]]))

                        biases['out'] = tf.Variable(tf.random_normal([1, n_classes]))

                        # needs to be adapted when different neuron counts, last one is always 1 and not the last element in the list
                        for i in range(len(hidden_neurons)):
                            outlist.append('Number of hidden neurons in layer ' + str(i) + ': ' + str(int(hidden_neurons[i])))
                            summ.append('Number of hidden neurons in layer' + str(i) + ': ' + str(int(hidden_neurons[i])))

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

                        ######################
                        # Construct model
                        pred = multilayer_perceptron(x, weights, biases, n_layers+2, drop_val)
                        pred_b = pred > 0.5

                        # Define loss and optimizer
                        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
                        # tf.summary.scalar("cost", cost)
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                        #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)

                        # accuracy measurement
                        correct_prediction = tf.equal(tf.cast(pred_b, "float"), y)
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
                        # tf.summary.scalar("accuracy", accuracy)

                        # Initializing the variables
                        init = tf.global_variables_initializer()

                        # stratified k-fold cross validation
                        skf = StratifiedKFold(n_splits=5)
                        #data_kv = []

                        #for i in range(len(Y)):
                        #    data_kv.append(np.append(X.iloc[i], Y[i]))

                        #data_kv = np.array(data_kv)

                        k_count = 1
                        accuracy_train = dict()
                        accuracy_test = dict()
                        sensitivity_dict = dict()
                        specifity_dict = dict()
                        fscores = []
                        fscores_weighted = []
                        # merge all tensorboard summaries
                        # merged = tf.summary.merge_all()

                        # Launch the graph for each fold
                        for k_train, k_test in skf.split(X, Y):

                            with tf.Session(
                                    config=config) as sess:  # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                                sess.run(init)

                                start = timeit.default_timer()
                                # Training cycle
                                cost_epoch_list = []
                                train_acc_epoch_list = []
                                test_acc_epoch_list = []



                                X_train, X_test = X[k_train], X[k_test]
                                Y_train, Y_test = Y[k_train], Y[k_test]


                                total_batch = int(len(Y_train) / batch_size)  # batch size is 9, 9 * 93 = 837

                                # if expression data, select new features on test set
                                if type == 'exp':
                                    selector = SelectKBest(f_classif, k=nfeat)
                                    X_train = selector.fit_transform(X_train, Y_train)
                                    X_test = selector.fit_transform(X_test, Y_test)

                                Y_train = Y_train.reshape(-1,1)
                                Y_test = Y_test.reshape(-1,1)
                                print(Y_train.shape)
                                for epoch in range(training_epochs):
                                    avg_cost = 0

                                    # Randomization of data
                                    X_train, Y_train = shuffle_train_data(X_train, Y_train)

                                    # split the array into batches
                                    X_batches = np.array_split(X_train, total_batch)
                                    Y_batches = np.array_split(Y_train, total_batch)

                                    # Loop over all batches
                                    for i in range(total_batch):  # print('batch x ', Y_batches[i])
                                        batch_x, batch_y = X_batches[i], Y_batches[i]
                                        batch_y.shape = (batch_y.shape[0], 1)

                                        # Run optimization op (backprop) and cost op (to get loss value)
                                        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                                        # summary_writer.add_summary(summary, epoch * total_batch + i)
                                        # outlist.append('Cost: ' + str(c) + 'kfold ' + str(k_count))

                                        # Compute average loss              ??? do we need this
                                        avg_cost += c / total_batch


                                    epoch_train_accuracy = accuracy.eval({x: X_train, y: Y_train, drop_prob:drop_val})
                                    epoch_test_accuracy = accuracy.eval({x: X_test, y: Y_test, drop_prob:drop_val})

                                    cost_epoch_list.append(avg_cost)
                                    train_acc_epoch_list.append(epoch_train_accuracy)
                                    test_acc_epoch_list.append(epoch_test_accuracy)

                                ###################################################################
                                ''''# plotting after every gridsearch
                                plt.figure(1)
                                graph_plot(211, range(1, len(cost_epoch_list) + 1), cost_epoch_list, 'cost')
                                title = 'Cost vs Epoch (Fold_' + str(k_count) + ')'
                                graph_legend(211, 'No. of epoch', 'Cost per epoch', title)

                                graph_plot(212, range(1, len(train_acc_epoch_list) + 1), train_acc_epoch_list, 'train accuracy')
                                graph_plot(212, range(1, len(test_acc_epoch_list) + 1), test_acc_epoch_list, 'test accuracy')
                                title = 'Accuracy vs Epoch (Fold_' + str(k_count) + ')'
                                graph_legend(212, 'No. of epoch', 'Accuracy', title)

                                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

                                epoch_file_name = "Fold_" + str(k_count) + ".png"
                                epoch_save_path = os.path.join(dir_path, epoch_file_name)
                                plt.savefig(epoch_save_path, format='png', dpi=1000, bbox_inches='tight')

                                epoch_file_name = "Fold_" + str(k_count) + ".svg"
                                epoch_save_path = os.path.join(dir_path, epoch_file_name)
                                plt.savefig(epoch_save_path, format='svg', dpi=1000, bbox_inches='tight')
                                plt.close()

                                plt.figure(2)
                                legend = 'Fold ' + str(k_count)
                                graph_plot(311, range(1, len(cost_epoch_list) + 1), cost_epoch_list, legend)'''

                                #####################
                                # Training accuracy
                                # X_train = new_x
                                # Y_train = new_y
                                # Y_train = np.array(Y_train)
                                # Y_train.shape = (Y_train.shape[0], 1)

                                # Test model on training data
                                print("Train Accuracy:", accuracy.eval({x: X_train, y: Y_train, drop_prob:1.0}))
                                train_accuracy = accuracy.eval({x: X_train, y: Y_train, drop_prob:1.0})
                                accuracy_train.update({k_count: train_accuracy})
                                #outlist.append('\n' + 'Fold: ' + str(k_count))
                                #outlist.append('Train Accuracy: ' + str(accuracy.eval({x: X_train, y: Y_train, drop_prob:1.0})))
                                # tf.summary.scalar('train acc', train_accuracy)

                                summ.append('\n' + 'Fold: ' + str(k_count))

                                summ.append('Train Accuracy: ' + str(accuracy.eval({x: X_train, y: Y_train, drop_prob:1.0})))
                                trainaccs.append(accuracy.eval({x: X_train, y: Y_train}))

                                # Train Sensitivity and Specifity
                                pred_labels = pred_b.eval({x: X_train, y: Y_train, drop_prob:1.0})
                                pred_labels = pred_labels.astype(int)
                                true_labels = Y_train
                                TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
                                TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
                                FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
                                FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

                                sensitivity = TP / (TP + FN)
                                specifity = TN / (TN + FP)

                                sensitivity_dict.update({k_count: sensitivity})
                                specifity_dict.update({k_count: specifity})

                                # Display sensitivity and specificity
                                print('Train sensitivity:', sensitivity)
                                print('Train specifity', specifity)
                                #outlist.append('Train Sensitivity: ' + str(sensitivity))
                                #outlist.append('Train Specificity: ' + str(specifity))
                                trainsens.append(sensitivity)
                                trainspec.append(specifity)

                                summ.append('Train Sensitivity: ' + str(sensitivity))
                                summ.append('Train Specificity: ' + str(specifity))

                                # Test Sensitivity and Specifity
                                pred_labels = pred_b.eval({x: X_test, y: Y_test, drop_prob: 1.0})
                                pred_labels = pred_labels.astype(int)
                                true_labels = Y_test
                                TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
                                TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
                                FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
                                FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

                                sensitivity = TP / (TP + FN)
                                specifity = TN / (TN + FP)

                                ########   adapt to test and train
                                sensitivity_dict.update({k_count: sensitivity})
                                specifity_dict.update({k_count: specifity})


                                # fscore for test
                                f = f1_score(true_labels, pred_labels, average=None)
                                fscores.append(f[1])

                                f_w = f1_score(true_labels, pred_labels, average='weighted')
                                fscores_weighted.append(f_w)


                                # Display sensitivity and specificity
                                print('Test sensitivity:', sensitivity)
                                print('Test specificity', specifity)
                                #outlist.append('Test Sensitivity: ' + str(sensitivity))
                                #outlist.append('Test Specificity: ' + str(specifity))
                                testsens.append(sensitivity)
                                testspec.append(specifity)

                                summ.append('Test Sensitivity: ' + str(sensitivity))
                                summ.append('Test Specificity: ' + str(specifity))
                                summ.append("F score: " + str(f[1]))

                                # Test model on test data

                                print("Test Accuracy:", accuracy.eval({x: X_test, y: Y_test, drop_prob:1.0}))
                                test_accuracy = accuracy.eval({x: X_test, y: Y_test, drop_prob:1.0})
                                accuracy_test.update({k_count: test_accuracy})
                                #outlist.append('Test Accuracy: ' + str(accuracy.eval({x: X_test, y: Y_test, drop_prob:1.0})))
                                k_count += 1

                                summ.append('Test Accuracy: ' + str(accuracy.eval({x: X_test, y: Y_test, drop_prob:1.0})))

                                testaccs.append(accuracy.eval({x: X_test, y: Y_test, drop_prob:1.0}))

                                # add weights and biases to report file
                                weights2, biases2 = sess.run([weights, biases])

                                for w in weights2:
                                    outlist.append(str(w) + ': ' + str(weights2[w].tolist()))

                                for b in biases2:
                                    outlist.append(str(b) + ': ' + str(biases2[b].tolist()))


                        avg_f = np.array(fscores).mean()
                        if avg_f > bestf:
                            bestf = avg_f
                            best_model_name = model_name
                            print("Found new best model:", avg_f)
                            ##### save all parameters needed
                        avg_fw = np.array(fscores_weighted).mean()
                        if avg_fw > best_wf:
                            best_wf = avg_fw
                            best_wf_model_name = model_name

                        new_f_score_dict.update({model_name:avg_f})
                        new_f_w_score_dict.update({model_name: avg_fw})

                        # adding statistics to summary file
                        values = [trainaccs, testaccs, trainsens, trainspec, testsens, testspec]
                        s = statistics(values)

                        summ.append("Train accuracy mean: " + str(s[0]))
                        summ.append("Train accuracy std: " + str(s[1]))
                        summ.append("Test accuracy mean: " + str(s[2]))
                        summ.append("Test accuracy std: " + str(s[3]))
                        summ.append("Train sensitivity mean: " + str(s[4]))
                        summ.append("Train sensitivity std: " + str(s[5]))
                        summ.append("Train specifity mean: " + str(s[6]))
                        summ.append("Train specifity std: " + str(s[7]))
                        summ.append("Test sensitivity mean: " + str(s[8]))
                        summ.append("Test sensitivity std: " + str(s[9]))
                        summ.append("Test specifity mean: " + str(s[10]))
                        summ.append("Test specifity std: " + str(s[11]) + "\n")

                        new_train_acc_dict.update({model_name:s[0]})
                        new_train_acc_std_dict.update({model_name:s[1]})
                        new_test_acc_dict.update({model_name:s[2]})
                        new_test_acc_std_dict.update({model_name:s[3]})
                        new_train_sens_dict.update({model_name:s[4]})
                        new_train_sens_std_dict.update({model_name:s[5]})
                        new_train_spec_dict.update({model_name:s[6]})
                        new_train_spec_std_dict.update({model_name:s[7]})
                        new_test_sens_dict.update({model_name:s[8]})
                        new_test_sens_std_dict.update({model_name:s[9]})
                        new_test_spec_dict.update({model_name:s[10]})
                        new_test_spec_std_dict.update({model_name:s[11]})




                        stop = timeit.default_timer()
                        #outlist.append('Trainig Time:' + str(stop - start))
                        summ.append('Trainig Time:' + str(stop - start))
                        ####################################################################################
                        # plotting for fold
                        '''plt.figure(2)

                        graph_legend(311, 'No. of epoch', 'Cost', 'Cost in each fold')

                        graph_plot(312, range(1, len(accuracy_train.keys()) + 1), accuracy_train.values(), 'Train Accuracy')
                        graph_plot(312, range(1, len(accuracy_test.keys()) + 1), accuracy_test.values(), 'Test Accuracy')
                        graph_legend(312, 'Folds', 'Accuracy', 'Train and Test Accuracy vs K-Folds')

                        graph_plot(313, range(1, len(accuracy_train.keys()) + 1), accuracy_train.values(), 'Train Accuracy')
                        graph_plot(313, range(1, len(accuracy_test.keys()) + 1), accuracy_test.values(), 'Test Accuracy')
                        graph_plot(313, range(1, len(sensitivity_dict.keys()) + 1), sensitivity_dict.values(), 'Sensitivity')
                        graph_plot(313, range(1, len(specifity_dict.keys()) + 1), specifity_dict.values(), 'Specifity')
                        graph_legend(313, 'Folds', 'Values', 'Sensitivity, Specifity, Accuracy vs Folds')

                        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

                        layer_file_name = "Summary_Figure_Layers_" + str(n_layers) + ".png"
                        layer_save_path = os.path.join(dir_path, layer_file_name)
                        plt.savefig(layer_save_path, format='png', dpi=1000, bbox_inches='tight')

                        layer_file_name = "Summary_Figure_Layers_" + str(n_layers) + ".svg"
                        layer_save_path = os.path.join(dir_path, layer_file_name)
                        plt.savefig(layer_save_path, format='svg', dpi=1000, bbox_inches='tight')
                        plt.close()

                        # graph_legend(211, 'No. of Epoch', 'Cost value', 'Cost value in each fold')

                        # write report file
                        with open(outputfile, 'w') as o:
                            for element in outlist:
                                o.write(element)
                                o.write("\n")

                        save_path = os.path.join(dir_path, outputfile_figure)
                        with open(save_path, 'w') as o:
                            for element in outlist:
                                o.write(element)
                                o.write(("\n"))'''

                        cv_time_stop = timeit.default_timer()
                        summ.append('Time for CV and Hidden layer trials: ' + str(cv_time_stop - cv_time_start))
                        summ.append('Time for CV and Hidden layer trials: ' + str(cv_time_stop - cv_time_start))
                        print("Time for CV and Hidden layer trials:", (cv_time_stop - cv_time_start))

    new_report_dict.update({'Train Accuracy:':new_train_acc_dict})
    new_report_dict.update({'Train Accuracy STD:':new_train_acc_std_dict})
    new_report_dict.update({'Test Accuracy:':new_test_acc_dict})
    new_report_dict.update({'Test Accuracy STD:':new_test_acc_std_dict})
    new_report_dict.update({'Train Sensitivity:': new_train_sens_dict})
    new_report_dict.update({'Train Sensitivity STD:': new_train_sens_std_dict})
    new_report_dict.update({'Train Specificity:': new_train_spec_dict})
    new_report_dict.update({'Train Specificity STD:':new_train_spec_std_dict})
    new_report_dict.update({'Test Sensitivity:':new_test_sens_dict})
    new_report_dict.update({'Test Sensitivity STD:': new_test_sens_std_dict})
    new_report_dict.update({'Test Specificity:': new_test_spec_dict})
    new_report_dict.update({'Test Specificity STD:': new_test_spec_std_dict})

    new_report_dict.update({'F1 Score:': new_f_score_dict})
    new_report_dict.update({'F1 Score Weighted:': new_f_w_score_dict})

    # write 1 summary file with only accuracies, sens, spec and statistics
    with open(summfile, 'w') as o:
        for element in summ:
            o.write(element)
            o.write('\n')

    #save_path = os.path.join(dir_path, summfile_figure)
    #with open(save_path, 'w') as o:
    #    for element in summ:
    #        o.write(element)
    #        o.write('\n')



    # for new file writing
    best_model_dict = []
    best_wf_model_dict = []
    with open(new_report_path,'w') as data:
        print('Best Model:  ')
        print('\n')
        for k1, v1 in new_report_dict.items():
            data.write(str(k1))
            data.write('\n')
            for k2 in v1:
                data.write(str(k2) + '\t' + str(v1[k2]) + '\n')
                if k2 == best_model_name:
                    print(str(k1) + '\t' + str(v1[k2]) + '\n')
                    best_model_dict.append(str(k1) + '\t' + str(v1[k2]))

                if k2 == best_wf_model_name:
                    best_wf_model_dict.append(str(k1) + '\t' + str(v1[k2]))


    with open(best_model_path, 'w') as data:
        data.write('For F1 Score:')
        data.write('\n')
        data.write(best_model_name)
        data.write('\n')
        for element in best_model_dict:
            data.write(element)
            data.write('\n')

        data.write('\n')
        data.write('for F1 Weighted Score:')
        data.write('\n')
        data.write(best_wf_model_name)
        data.write('\n')
        for element in best_wf_model_dict:
            data.write(element)
            data.write('\n')