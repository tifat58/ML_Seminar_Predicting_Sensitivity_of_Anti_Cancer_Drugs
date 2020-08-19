from __future__ import print_function

import pickle
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import defaultdict
# import matplotlib.pyplot as plt
# from tkinter import *
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib
import timeit
matplotlib.use('agg')
import matplotlib.pyplot as plt

# GPU usage
gpu = sys.argv[3]
# type of feature file
type = sys.argv[6]

# for ix in [1]:
with tf.device('/device:GPU:' + str(gpu)):
    # 1 argument: number of features
    # 2 argument: which drug
    # 3 argument: GPU
    # 4. argument: lower number of total layers
    # 5. argument: upper number of total layers
    # 6. argument: type of features

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

    def statistics(tracc, teacc, sens, spec):
        """computes mean and std for different sets"""
        tr = np.array(tracc)
        te = np.array(teacc)
        se = np.array(sens)
        sp = np.array(spec)

        infos = []

        # train acc, test acc, senseitivity, specificity
        # first mean, then std
        infos.append(np.mean(tr))
        infos.append(np.std(tr))
        infos.append(np.mean(te))
        infos.append(np.std(te))
        infos.append(np.mean(se))
        infos.append(np.std(se))
        infos.append(np.mean(sp))
        infos.append(np.std(sp))

        return infos


    # Create model with x as featues, weights, biases and total number of layers
    def multilayer_perceptron(x, weights, biases, n_hidden):
        print("x.shape", x.shape)
        print("weights", weights)

        # Hidden layer with RELU activation : changed to sigmoid
        previous = x

        for i in range(1, n_hidden - 1):
            activ1 = tf.add(tf.matmul(previous, weights['w' + str(i)]), biases['b' + str(i)])
            activ1 = tf.nn.dropout(activ1, keep_prob=1.0)
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
    #    feat = pd.read_excel('../data/expression_final3.xlsx')
    #    feat = feat.T[1:]
    #    with open('exp.file', 'wb') as f:
    #        pickle.dump(feat, f, pickle.HIGHEST_PROTOCOL)
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
    print(indexf)
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
    # nfeat = len(feat.columns)
    # print("after dropout: " + str(nfeat))

    ###############
    # specify model parameters
    learning_rate = 0.01 #Modified
    training_epochs = 400 #Modified
    batch_size = 30
    display_step = 100
    # logs_path = '../log_neural_network_' + sys.argv[2]
    # outputfile = "report_" + sys.argv[2] + "_" + str(nfeat) + "_cnv.txt"

    # Network Parameters
    n_input = nfeat  # Number of feature
    n_classes = 1  # Number of classes to predict

    #################
    # session configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allocator_type = "BFC"
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True

    # for overview file
    summ = list()
    stats = []
    summfile = "../reports/summary_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep400_ba30_exp_lr_" + str(learning_rate) + '.txt'
    summfile_figure = "summary_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep400_ba30_exp_lr_" + str(learning_rate) + '.txt'

    # Tensorflow placeholders for features and repsonse
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, None])

    # create saver
    # saver = tf.train.Saver()
    cv_time_start = timeit.default_timer()
    # gridsearch on number of hidden layers
    # 4 lower bound, 5 upper bound
    for n_layers in range(int(sys.argv[4]), int(sys.argv[5]) + 1):

        # logarthmic stepsize of number of hidden neurons
        hidden_neurons = np.logspace(np.log10(float(sys.argv[1])), np.log10(5.0), num=n_layers)

        # initialization of report lists
        outlist = list()
        trainaccs = []
        sens = []
        spec = []
        testaccs = []

        ################################################################
        # creating folders to store plot for each of the grid search
        if not os.path.exists('../Figure'):
            os.mkdir('../Figure')

        dir_path = "../Figure/" + "drug_" + str(sys.argv[2]) + "_features_" + str(sys.argv[1]) + "_epoch_" + str(
            training_epochs) + "_nLayers_" + str(n_layers) + "_" + str(sys.argv[6]) + 'lr_' + str(learning_rate)

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
        outputfile = "../reports/report_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep500_ba30_" + str(
            n_layers) + "_exp_lr_" + str(learning_rate) + ".txt"
        outputfile_figure = "report_" + sys.argv[2] + "_" + sys.argv[1] + "_layerslog5_ep500_ba30_" + str(
            n_layers) + "_exp_lr_" + str(learning_rate) + ".txt"
        # summ.append('Number of layers: ' + str(n_layers))
        outlist.append('Number hidden layers: ' + str(n_layers))
        summ.append('Number hidden layers: ' + str(n_layers))

        # weight and bias setting: last one is set afterwards
        weights = {}
        for i in range(1, (n_layers - 1)):
            # print(np.floor(hidden_neurons[i-1]))
            weights['w' + str(i)] = tf.Variable(
                tf.random_normal([int(np.around(hidden_neurons[i - 1])), int(np.around(hidden_neurons[i]))]))

        weights['out'] = tf.Variable(tf.random_normal([int(np.around(hidden_neurons[-2])), n_classes]))

        biases = {}
        for i in range(1, n_layers - 1):
            biases['b' + str(i)] = tf.Variable(tf.random_normal([1, int(np.around(hidden_neurons[i]))]))

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
        pred = multilayer_perceptron(x, weights, biases, n_layers)
        pred_b = pred > 0.5

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
        # tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        # accuracy measurement
        correct_prediction = tf.equal(tf.cast(pred_b, "float"), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        # tf.summary.scalar("accuracy", accuracy)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # k-fold cross validation
        data_kv = []
        # if len(sys.argv) > 2:
        for i in range(len(Y)):
            data_kv.append(np.append(X.iloc[i], Y[i]))
        #    data_kv.append(np.append(X[X.columns[i]], Y[i]))
        # else:
        #    for i in range(len(Y)):
        #        data_kv.append(np.append(X[i], Y[i]))
        data_kv = np.array(data_kv)
        # print('data kv ', data_kv)
        # data_kv = np.append(X, Y, axis=1)
        # print(data_kv.shape, type(data_kv))
        kf = KFold(n_splits=10)

        k_count = 1
        accuracy_train = dict()
        accuracy_test = dict()
        sensitivity_dict = dict()
        specifity_dict = dict()
        # merge all tensorboard summaries
        # merged = tf.summary.merge_all()

        # Launch the graph for each fold
        for k_train, k_test in kf.split(data_kv):

            with tf.Session(
                    config=config) as sess:  # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                sess.run(init)
                # summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
                saver = tf.train.Saver()
                start = timeit.default_timer()
                # Training cycle
                cost_epoch_list = []
                train_acc_epoch_list = []
                test_acc_epoch_list = []

                total_batch = int(len(data_kv[k_train]) / batch_size)  # batch size is 9, 9 * 93 = 837
                new_y = []
                new_x = []
                for i in data_kv[k_train]:
                    new_x.append(i[:-1])
                    new_y.append(i[-1])
                # print('newx ', len(new_x))
                # print('newy ', len(new_y))

                # if expression data, select new features on test set
                if type == 'exp':
                    selector = SelectKBest(f_classif, k=nfeat)
                    new_x = selector.fit_transform(new_x, new_y)

                # data for train
                # Training accuracy
                X_train = new_x
                Y_train = new_y
                Y_train = np.array(Y_train)
                Y_train.shape = (Y_train.shape[0], 1)

                # for test
                new_test_y = []
                new_test_x = []
                for i in data_kv[k_test]:
                    new_test_x.append(i[:-1])
                    new_test_y.append(i[-1])

                if type == 'exp':
                    selector = SelectKBest(f_classif, k=nfeat)
                    new_test_x = selector.fit_transform(new_test_x, new_test_y)

                new_test_y = np.array(new_test_y)
                new_test_y.shape = (new_test_y.shape[0], 1)

                for epoch in range(training_epochs):
                    avg_cost = 0

                    # Randomization of data
                    new_x, new_y = shuffle_train_data(np.asarray(new_x), np.asarray(new_y))

                    # split the array into batches
                    X_batches = np.array_split(new_x, total_batch)
                    Y_batches = np.array_split(new_y, total_batch)

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

                    print()
                    epoch_train_accuracy = accuracy.eval({x: X_train, y: Y_train})
                    epoch_test_accuracy = accuracy.eval({x: new_test_x, y: new_test_y})

                    cost_epoch_list.append(avg_cost)
                    train_acc_epoch_list.append(epoch_train_accuracy)
                    test_acc_epoch_list.append(epoch_test_accuracy)

                ###################################################################
                # plotting after every gridsearch
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
                graph_plot(311, range(1, len(cost_epoch_list) + 1), cost_epoch_list, legend)

                #####################
                # Training accuracy
                # X_train = new_x
                # Y_train = new_y
                # Y_train = np.array(Y_train)
                # Y_train.shape = (Y_train.shape[0], 1)

                # Test model on training data
                print("Train Accuracy:", accuracy.eval({x: X_train, y: Y_train}))
                train_accuracy = accuracy.eval({x: X_train, y: Y_train})
                accuracy_train.update({k_count: train_accuracy})
                outlist.append('\n' + 'Fold: ' + str(k_count))
                outlist.append('Train Accuracy: ' + str(accuracy.eval({x: X_train, y: Y_train})))
                # tf.summary.scalar('train acc', train_accuracy)

                summ.append('\n' + 'Fold: ' + str(k_count))

                summ.append('Train Accuracy: ' + str(accuracy.eval({x: X_train, y: Y_train})))
                trainaccs.append(accuracy.eval({x: X_train, y: Y_train}))

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

                sensitivity_dict.update({k_count: sensitivity})
                specifity_dict.update({k_count: specifity})

                # Display sensitivity and specificity
                print('sensitivity:', sensitivity)
                print('specifity', specifity)
                outlist.append('Sensitivity: ' + str(sensitivity))
                outlist.append('Specificity: ' + str(specifity))
                sens.append(sensitivity)
                spec.append(specifity)

                summ.append('Sensitivity: ' + str(sensitivity))
                summ.append('Specificity: ' + str(specifity))

                # Test model on test data

                print("Test Accuracy:", accuracy.eval({x: new_test_x, y: new_test_y}))
                test_accuracy = accuracy.eval({x: new_test_x, y: new_test_y})
                accuracy_test.update({k_count: test_accuracy})
                outlist.append('Test Accuracy: ' + str(accuracy.eval({x: new_test_x, y: new_test_y})))
                k_count += 1

                summ.append('Test Accuracy: ' + str(accuracy.eval({x: new_test_x, y: new_test_y})))

                testaccs.append(accuracy.eval({x: new_test_x, y: new_test_y}))

                # add weights and biases to report file
                weights2, biases2 = sess.run([weights, biases])

                for w in weights2:
                    outlist.append(str(w) + ': ' + str(weights2[w].tolist()))

                for b in biases2:
                    outlist.append(str(b) + ': ' + str(biases2[b].tolist()))

                modelpath = 'best_' + sys.argv[1] + '_' + str(k_count - 1) + '.ckpt'
                if accuracy.eval({x: new_test_x, y: new_test_y}) > 80 and accuracy.eval(
                        {x: X_train, y: Y_train}) > 80 and accuracy.eval({x: X_train, y: Y_train}) > accuracy.eval(
                    {x: new_test_x, y: new_test_y}) and sensitivity > 55 and specifity > 95:
                    saver.save(sess, modelpath)

            # adding statistics to summary file
            s = statistics(trainaccs, testaccs, sens, spec)

            summ.append("Train acc mean: " + str(s[0]))
            summ.append("Train acc std: " + str(s[1]))
            summ.append("Test acc mean: " + str(s[2]))
            summ.append("Test acc std: " + str(s[3]))
            summ.append("Sens mean: " + str(s[4]))
            summ.append("Sens std: " + str(s[5]))
            summ.append("Spec mean: " + str(s[6]))
            summ.append("Spec std: " + str(s[7]) + "\n")

        stop = timeit.default_timer()
        outlist.append('Trainig Time:' + str(stop - start))
        summ.append('Trainig Time:' + str(stop - start))
        ####################################################################################
        # plotting for fold
        plt.figure(2)

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
                o.write(("\n"))

        cv_time_stop = timeit.default_timer()
        summ.append('Time for CV and Hidden layer trials: ' + str(cv_time_stop - cv_time_start))
        summ.append('Time for CV and Hidden layer trials: ' + str(cv_time_stop - cv_time_start))
        print("Time for CV and Hidden layer trials:", (cv_time_stop - cv_time_start))
    # write 1 summary file with only accuracies, sens, spec and statistics
    with open(summfile, 'w') as o:
        for element in summ:
            o.write(element)
            o.write('\n')

    save_path = os.path.join(dir_path, summfile_figure)
    with open(save_path, 'w') as o:
        for element in summ:
            o.write(element)
            o.write('\n')

