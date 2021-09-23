import string
from os import listdir
import numpy as np
import math as math
import sys
from str2bool import str2bool
np.set_printoptions(threshold=sys.maxsize)

flag_NB = str2bool(sys.argv[1])
flag_LR = str2bool(sys.argv[2])
flag_PTR = str2bool(sys.argv[3])
flag_HP = str2bool(sys.argv[4])
train_set = sys.argv[5]
test_set = sys.argv[6]


def main():  # main function that runs the file
    train_dir = listdir(train_set)  # set your training directory
    test_dir = listdir(test_set)  # set your test directory
    h_dict = dict()  # init an empty ham training dictionary
    s_dict = dict()  # init an empty spam training dictionary
    dictionary_builder(train_dir, h_dict, s_dict)  # build the ham and spam dictionaries
    m = get_m(h_dict, s_dict)  # find the number of unique values in the training set
    #test
    # Naive-Bayes
    if flag_NB:

        h_prob, s_prob = prob_ham_spam(train_dir)  # get Pr[y], or the probability that any given email is ham/spam
        predictions = prob_finder(m, h_dict, s_dict, test_dir, h_prob, s_prob)  # create a dictionary of predictions
        acc = get_accuracy(predictions)  # get the accuracy of our predictions
        print("NB test set accuracy ", acc, sep="\t")

    # End Naive Bayes

    # MCAP Logisitic Regression
    if flag_LR:

        y_dict = dict()
        y_dict["ham"] = 1
        y_dict["spam"] = 0
        trainh_dict = dict()
        trains_dict = dict()
        valh_dir = []
        vals_dir = []
        LR_dictionary_builder(train_dir, trainh_dict, trains_dict, valh_dir, vals_dir)
        ham_dir = train_set + "ham"
        spam_dir = train_set + "spam"
        w0 = 1
        lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
        best_lambda = 0
        best_acc = 0
        if flag_HP:
            m = get_m(trainh_dict, trains_dict)
            tr_h_70, v_h_30, tr_s_70, tr_s_30 = split_dir_70_30(listdir(ham_dir), listdir(spam_dir))
            x, y, vocab, weights = init_weights(m, tr_h_70, tr_s_70, trainh_dict, trains_dict)
            valid_x = X_builder(ham_dir, spam_dir, vocab, m,True)
            print("lambda", "w0", "accuracy", sep="\t")
            for lambduh in lambdas:
                weights = np.zeros((len(weights), 1))
                new_w = LR_weight_training(w0, x, y, vocab, weights, lambduh)
                valid_preds = LR_validate_weights(w0, new_w, ham_dir, spam_dir, vocab, m, valid_x)
                print(lambduh, w0,  get_accuracy(valid_preds), sep="\t")
                if get_accuracy(valid_preds) >= best_acc:
                    best_lambda = lambduh
                    best_acc = get_accuracy(valid_preds)
        print("LR ideal lambda ", best_lambda, sep="\t")
        lambduh = best_lambda
        m = get_m(h_dict, s_dict)
        x, y, vocab, weights = init_weights(m, listdir(ham_dir), listdir(spam_dir), h_dict, s_dict)
        new_w = LR_weight_training(w0, x, y, vocab, weights, lambduh)
        test_ham_dir = test_set + 'ham'
        test_spam_dir = test_set + 'spam'
        test_x = X_builder(test_ham_dir, test_spam_dir, vocab, m, False)
        new_preds = LR_test_files(w0, new_w, test_ham_dir, test_spam_dir, test_x)
        print("LR test set accuracy ", get_accuracy(new_preds), sep="\t")

    # End MCAP Logisitic Regression

    if flag_PTR:

        test_ham_dir = test_set + 'ham'
        test_spam_dir = test_set + 'spam'
        trainh_dict = dict()
        trains_dict = dict()
        valh_dir = []
        vals_dir = []
        LR_dictionary_builder(train_dir, trainh_dict, trains_dict, valh_dir, vals_dir)
        ham_dir = train_set + "ham"
        spam_dir = train_set + "spam"
        w0 = 1
        best_itr = 0
        best_acc = 0
        print("itrs", "w0", "accuracy", sep="\t")
        if flag_HP:
            m = get_m(trainh_dict, trains_dict)
            tr_h_70, v_h_30, tr_s_70, tr_s_30 = split_dir_70_30(listdir(ham_dir), listdir(spam_dir))
            # train set
            x, y, vocab, weights = init_PTR_weights(m, tr_h_70, tr_s_70, trainh_dict, trains_dict)
            valid_x = X_builder(ham_dir, spam_dir, vocab, m, True)
            iterations = [1,5,10,15,20,25]
            for iteration in iterations:
                new_w = PTR_updates(x, y, weights, w0, iteration)
                valid_preds = PTR_tester(valid_x, new_w, w0, ham_dir, spam_dir, True)
                print(iteration, w0, get_accuracy(valid_preds), sep="\t")
                if get_accuracy(valid_preds) > best_acc:
                    best_itr = iteration
                    best_acc = get_accuracy(valid_preds)

        # test set
        print("PTR ideal iterations ", best_itr, sep="\t")
        iterations = best_itr
        m = get_m(h_dict, s_dict)
        x, y, vocab, weights = init_PTR_weights(m, listdir(ham_dir), listdir(spam_dir), h_dict, s_dict)
        new_w = PTR_updates(x, y, weights, w0, iterations)
        test_x = X_builder(test_ham_dir, test_spam_dir, vocab, m, False)
        new_preds = PTR_tester(test_x, new_w, w0, test_ham_dir, test_spam_dir, False)
        print("PTR test set accuracy ", get_accuracy(new_preds), sep="\t")

    # End Perceptron Training Rule

def PTR_updates(x, y, input_w, w0, n):
    eta = 0.01
    iteration = 0
    w = input_w
    while iteration < n:
        output = np.sign(np.dot(x, w) + w0)
        inner = y - output
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                w[j, 0] = w[j, 0] + eta*inner[i, 0]*x[i, j]
        iteration += 1
    return w

def PTR_tester(x, w, w0, ham_dir, spam_dir, flag):
    output = np.sign(np.dot(x, w) + w0)
    h_dir = listdir(ham_dir)
    s_dir = listdir(spam_dir)
    predictions = dict()
    if flag:
        n30_ham_dir = math.floor(len(h_dir) * 0.7)
        n30_spam_dir = math.floor(len(s_dir) * 0.7)
        a = h_dir[int(n30_ham_dir):len(h_dir)]
        b = s_dir[int(n30_spam_dir):len(s_dir)]
        files = np.concatenate((a, b))
        for i in range(len(output)):
            if output[i, 0] == 1:
                predictions[files[i]] = 'spam'
            else:
                predictions[files[i]] = 'ham'
    else:
        a = h_dir[0:len(h_dir)]
        b = s_dir[0:len(s_dir)]
        files = np.concatenate((a, b))
        for i in range(len(output)):
            if output[i, 0] == 1:
                predictions[files[i]] = 'spam'
            else:
                predictions[files[i]] = 'ham'
    return predictions


def X_builder(ham_dir, spam_dir, vocab, m, flag):
    h_dir = listdir(ham_dir)
    s_dir = listdir(spam_dir)
    n30_ham_dir = math.floor(len(h_dir) * 0.7)
    n30_spam_dir = math.floor(len(s_dir) * 0.7)
    if flag:
        new_x = np.zeros((len(h_dir) - n30_ham_dir + len(s_dir) - n30_spam_dir, m))
        for file in range(int(n30_ham_dir), int(len(h_dir))):
            this_file = h_dir[file]
            content = open(ham_dir + '/' + this_file, 'r', encoding='latin-1')
            for line in content:  # clean each line
                line = line.strip()  # string leading/trailing spaces
                line = line.lower()  # clean to all lower case
                line = line.translate(line.maketrans("", "", string.punctuation))  # strip punctuation
                line = ' '.join(line.split())  # strip excess spaces
                text = line.split(' ')  # split each line into a list of words
                for word in text:  # get the counts for each word in the file
                    if not word.isnumeric():  # do not allow arabic numerals
                        if word in vocab:  # add to word counter if it exists in the dictionary
                            position = vocab.index(word)
                            new_x[file - int(n30_ham_dir), position] = new_x[file - int(n30_ham_dir), position] + 1
        for file in range(int(n30_spam_dir), int(len(s_dir))):
            this_file = s_dir[file]
            content = open(spam_dir + '/' + this_file, 'r', encoding='latin-1')
            for line in content:  # clean each line
                line = line.strip()  # string leading/trailing spaces
                line = line.lower()  # clean to all lower case
                line = line.translate(line.maketrans("", "", string.punctuation))  # strip punctuation
                line = ' '.join(line.split())  # strip excess spaces
                text = line.split(' ')  # split each line into a list of words
                for word in text:  # get the counts for each word in the file
                    if not word.isnumeric():  # do not allow arabic numerals
                        if word in vocab:  # add to word counter if it exists in the dictionary
                            position = vocab.index(word)
                            new_x[file + (len(h_dir) - n30_ham_dir) - n30_spam_dir, position] = new_x[file + (len(h_dir) - n30_ham_dir) - n30_spam_dir, position] + 1
    else:
        new_x = np.zeros((len(h_dir) + len(s_dir), m))
        for file in range(0, int(len(h_dir))):
            this_file = h_dir[file]
            content = open(ham_dir + '/' + this_file, 'r', encoding='latin-1')
            for line in content:  # clean each line
                line = line.strip()  # string leading/trailing spaces
                line = line.lower()  # clean to all lower case
                line = line.translate(line.maketrans("", "", string.punctuation))  # strip punctuation
                line = ' '.join(line.split())  # strip excess spaces
                text = line.split(' ')  # split each line into a list of words
                for word in text:  # get the counts for each word in the file
                    if not word.isnumeric():  # do not allow arabic numerals
                        if word in vocab:  # add to word counter if it exists in the dictionary
                            position = vocab.index(word)
                            new_x[file, position] = new_x[file, position] + 1
        for file in range(0, int(len(s_dir))):
            this_file = s_dir[file]
            content = open(spam_dir + '/' + this_file, 'r', encoding='latin-1')
            for line in content:  # clean each line
                line = line.strip()  # string leading/trailing spaces
                line = line.lower()  # clean to all lower case
                line = line.translate(line.maketrans("", "", string.punctuation))  # strip punctuation
                line = ' '.join(line.split())  # strip excess spaces
                text = line.split(' ')  # split each line into a list of words
                for word in text:  # get the counts for each word in the file
                    if not word.isnumeric():  # do not allow arabic numerals
                        if word in vocab:  # add to word counter if it exists in the dictionary
                            position = vocab.index(word)
                            new_x[file + len(h_dir), position] = new_x[file + len(h_dir), position] + 1
    return new_x

# function name: validate weights
# arguments: bias w0, weights dictionary, ham and spam validation directories
# purpose: calculates the accuracy of the trained model against the validation set
# returns: the accuracy?????????????????????????????????????????
def LR_validate_weights(w0, weights, ham_dir, spam_dir, vocab, m, new_x):
    h_dir = listdir(ham_dir)
    s_dir = listdir(spam_dir)
    n30_ham_dir = math.floor(len(h_dir) * 0.7)
    predictions = dict()
    n30_spam_dir = math.floor(len(s_dir) * 0.7)
    a = h_dir[int(n30_ham_dir):len(h_dir)]
    b = s_dir[int(n30_spam_dir):len(s_dir)]
    files = np.concatenate((a, b))
    these_preds = sigmoid(np.dot(new_x, weights) + w0)

    for i in range(len(these_preds)):
        this_pred = these_preds[i, 0]
        this_pred = int(round(this_pred))
        if this_pred == 1:
            predictions[files[i]] = 'spam'
        else:
            predictions[files[i]] = 'ham'
    return predictions


def LR_test_files(w0, weights, ham_dir, spam_dir, test_x):
    h_dir = listdir(ham_dir)
    s_dir = listdir(spam_dir)
    predictions = dict()
    a = h_dir[0:len(h_dir)]
    b = s_dir[0:len(s_dir)]
    files = np.concatenate((a, b))
    these_preds = sigmoid(np.dot(test_x, weights) + w0)
    for i in range(len(these_preds)):
        this_pred = these_preds[i, 0]
        this_pred = int(round(this_pred))
        if this_pred == 1:
            predictions[files[i]] = 'spam'
        else:
            predictions[files[i]] = 'ham'
    return predictions

# function name: LR_weight_training
def LR_weight_training(w0, x, y, vocab, weights, lambduh):
    eta = 0.0001
    iterations = 1000
    for n in range(0, iterations):
        w_trans_x = np.dot(x, weights) + w0
        sig_inner = sigmoid(w_trans_x)
        del_l_del_w = np.dot(x.transpose(), (y-sig_inner))
        weights = weights + eta*del_l_del_w - eta*lambduh*weights
    return weights


def sigmoid(array):
    sig = np.divide(np.exp(array), (1+np.exp(array)))
    return sig

# function name: LR_dictionary_builder
# arguments: the training directory, two empty (ham, spam) dictionaries, two empty (ham, spam) directories
# purpose: performs the 70/30 split, creates the ham and spam dictionaries with the 70, makes directories for the
# validation sets to be processed later on
# returns: none, performs all functions in-place
def LR_dictionary_builder(my_dir, trainh_dict, trains_dict, valh_dir, vals_dir):
    for directory in my_dir:  # in set ['ham', 'spam']
        this_dir = listdir(train_set + directory)  # list of files in ham or spam folder
        num_files = len(this_dir)
        num_set70 = np.floor(num_files * 0.70)
        for i in range(0, int(num_set70)):
            content = open(train_set + directory + '/' + this_dir[i], 'r', encoding='latin-1')  # open each file
            for line in content:  # look at each line
                line = line.strip()  # strip leading/trailing spaces
                line = line.lower()  # convert the contents to lower case characters
                line = line.translate(
                    line.maketrans("", "", string.punctuation))  # strip the punctuation (per question 22 on canvas)
                line = ' '.join(line.split())  # remove excess spaces
                text = line.split(' ')  # split into a list of words
                for word in text:  # check each word in the line
                    if not word.isnumeric():  # do not include arabic numerals
                        if directory == 'ham':  # if this is a ham training set
                            if word in trainh_dict:  # if it's already in the dictionary, increase the count
                                trainh_dict[word] = trainh_dict[word] + 1
                            else:  # otherwise add a new entry to the dictionary
                                trainh_dict[word] = 1
                        else:  # if it's in the spam training set
                            if word in trains_dict:  # if it's already in the spam dictionary, increase count of word
                                trains_dict[word] = trains_dict[word] + 1
                            else:  # otherwise add new entry
                                trains_dict[word] = 1

        for j in range(int(num_set70), int(num_files)):
            if directory == 'ham':
                valh_dir.append(train_set + directory + '/' + this_dir[j])
            if directory == 'spam':
                vals_dir.append(train_set + directory + '/' + this_dir[j])

def split_dir_70_30(h_dir, s_dir):
    new_h_dir_70 = h_dir[0:math.floor(len(h_dir)*0.7)]
    new_h_dir_30 = h_dir[math.floor(len(h_dir)*0.7):len(h_dir)]
    new_s_dir_70 = s_dir[0:math.floor(len(s_dir) * 0.7)]
    new_s_dir_30 = s_dir[math.floor(len(s_dir) * 0.7):len(s_dir)]
    return new_h_dir_70, new_h_dir_30, new_s_dir_70, new_s_dir_30

def build_dict_for_file(file):
    d = dict()
    content = open(file, 'r', encoding='latin-1')  # this file's contents
    for line in content:  # clean each line
        line = line.strip()  # string leading/trailing spaces
        line = line.lower()  # clean to all lower case
        line = line.translate(line.maketrans("", "", string.punctuation))  # strip punctuation
        line = ' '.join(line.split())  # strip excess spaces
        text = line.split(' ')  # split each line into a list of words
        for word in text:  # get the counts for each word in the file
            if not word.isnumeric():  # do not allow arabic numerals
                if word in d:  # add to word counter if it exists in the dictionary
                    d[word] = d[word] + 1
                else:  # otherwise add as a new key
                    d[word] = 1
    return d

# function name: init_weights
# arguments: a ham dictionary and a spam dictionary
# purpose: creates the dictionary of weight values at initial values
# returns: the weights dictionary
def init_weights(m, h_dir, s_dir, h_dict, s_dict):
    n = len(h_dir)+len(s_dir)
    x = np.zeros((n, m))
    vocab = []
    for file in range(0, len(h_dir)):

        this_file_h_dict = build_dict_for_file(train_set + "/ham/" + h_dir[file])
        for key in list(this_file_h_dict.keys()):
            if key not in vocab:
                vocab.append(key)
                pos = vocab.index(key)
                x[file, pos] = x[file, pos] + this_file_h_dict[key]
            else:
                pos = vocab.index(key)
                x[file, pos] = x[file, pos] + this_file_h_dict[key]
    for file in range(0, len(s_dir)):
        this_file_s_dict = build_dict_for_file(train_set + "/spam/" + s_dir[file])
        for key in list(this_file_s_dict.keys()):
            if key not in vocab:
                vocab.append(key)
                pos = vocab.index(key)
                x[file+len(h_dir), pos] = x[file+len(h_dir), pos] + this_file_s_dict[key]
            else:
                pos = vocab.index(key)
                x[file+len(h_dir), pos] = x[file+len(h_dir), pos] + this_file_s_dict[key]
    y = np.zeros((n, 1))
    y[n-len(s_dir):n, :] = 1
    weights = np.zeros((m, 1))
    return x, y, vocab, weights  # return the weights dictionary

def init_PTR_weights(m, h_dir, s_dir, h_dict, s_dict):
    n = len(h_dir)+len(s_dir)
    x = np.zeros((n, m))
    vocab = []
    for file in range(0, len(h_dir)):
        this_file_h_dict = build_dict_for_file(train_set + "/ham/" + h_dir[file])
        for key in list(this_file_h_dict.keys()):
            if key not in vocab:
                vocab.append(key)
                pos = vocab.index(key)
                x[file, pos] = x[file, pos] + this_file_h_dict[key]
            else:
                pos = vocab.index(key)
                x[file, pos] = x[file, pos] + this_file_h_dict[key]
    for file in range(0, len(s_dir)):
        this_file_s_dict = build_dict_for_file(train_set + "/spam/" + s_dir[file])
        for key in list(this_file_s_dict.keys()):
            if key not in vocab:
                vocab.append(key)
                pos = vocab.index(key)
                x[file+len(h_dir), pos] = x[file+len(h_dir), pos] + this_file_s_dict[key]
            else:
                pos = vocab.index(key)
                x[file+len(h_dir), pos] = x[file+len(h_dir), pos] + this_file_s_dict[key]
    y = np.zeros((n, 1)) - 1
    y[n-len(s_dir):n, :] = 1
    weights = np.zeros((m, 1))
    return x, y, vocab, weights  # return the weights dictionary

# function name: dictionary_builder
# arguments: the training directory, the empty ham and spam dictionaries
# purpose: fills out the ham and spam dictionaries with unique words and counts based on the training data
# returns: none, fills the dictionaries within the function
def dictionary_builder(my_dir, ham_dict, spam_dict):
    for directory in my_dir:  # in set ['ham', 'spam']
        this_dir = listdir(train_set + directory)  # list of files in ham or spam folder
        for file in this_dir:
            content = open(train_set + directory + '/' + file, 'r', encoding='latin-1')  # open each file
            for line in content:  # look at each line
                line = line.strip()  # strip leading/trailing spaces
                line = line.lower()  # convert the contents to lower case characters
                line = line.translate(
                    line.maketrans("", "", string.punctuation))  # strip the punctuation (per question 22 on canvas)
                line = ' '.join(line.split())  # remove excess spaces
                text = line.split(' ')  # split into a list of words
                for word in text:  # check each word in the line
                    if not word.isnumeric():  # do not include arabic numerals
                        if directory == 'ham':  # if this is a ham training set
                            if word in ham_dict:  # if it's already in the dictionary, increase the count
                                ham_dict[word] = ham_dict[word] + 1
                            else:  # otherwise add a new entry to the dictionary
                                ham_dict[word] = 1
                        else:  # if it's in the spam training set
                            if word in spam_dict:  # if it's already in the spam dictionary, increase count of word
                                spam_dict[word] = spam_dict[word] + 1
                            else:  # otherwise add new entry
                                spam_dict[word] = 1


# function name: word_counter
# arguments: a dictionary
# purpose: find the number of total words in a dictionary
# returns: integer count of total words in a dictionary
def word_counter(dictionary):
    counter = 0  # init as 0
    for key in list(dictionary.keys()):  # iterate through keys
        counter = counter + dictionary[key]  # add value to counter
    return counter


# function name: print_dict
# arguments: a dictionary
# purpose: prints a dictionary to console. not used in actual functions but helpful for debugging
# returns: none
def print_dict(dictionary):
    for key in list(dictionary.keys()):  # iterate through the keys
        print(key, ":", dictionary[key])  # print the key and it's value


# function name: get_m
# arguments: the ham and spam dictionaries
# purpose: find the total number of unique words in the training data (or between any two dictionaries)
# returns: an integer m representing the number of unique words
def get_m(ham_dict, spam_dict):
    unique_list = []  # init an empty list of unique words
    for key in ham_dict:  # iterate through the ham dict
        if key not in unique_list:  # if it isn't in the list
            unique_list.append(key)  # add to list
    for key in spam_dict:  # iterate through the spam dict
        if key not in unique_list:  # if it isn't in the list
            unique_list.append(key)  # add to list
    m = len(unique_list)  # m is number of unique words
    return m


# function name: prob_ham_spam
# arguments: the training directory list (in the set [ham,spam])
# purpose: find Pr[y], or the class balance in the training data
# returns: two floats: first the probability of ham, second, the prob of spam (should add up to 1)
def prob_ham_spam(my_dir):
    counter = 0  # init empty counter variable
    for directory in my_dir:  # iterate through the ham and spam folders
        this_dir = listdir(train_set + directory)  # list of files in this folder
        counter = counter + len(this_dir)  # add the number of files to the counter
        if directory == 'ham':  # if this directory is ham, save the number of files for later
            h_counter = len(this_dir)
    ham_prob = h_counter / counter  # share of files that were ham
    spam_prob = (counter - h_counter) / counter  # share of files that were spam
    return ham_prob, spam_prob


# function name: get_file_prob
# arguments: a file, a boolean flag (True = getting the ham probability), the directory, the ham dictionary from
# training, the spam dictionary from training, the Pr[y]s as ham/spam_prob, and the number of unique values m
# purpose: find the probability that a given file is ham or spam depending on the boolean flag (in ln representation) by
# adding the natural logs of individual probabilities
# returns: a float equal to the natural log of the probability that the file belongs to a class
def get_file_prob(file, flag, directory, ham_dict, spam_dict, ham_prob, spam_prob, m):
    d = dict()  # dictionary for this file at this directory (for word counting)
    content = open(test_set + directory + '/' + file, 'r', encoding='latin-1')  # this file's contents
    for line in content:  # clean each line
        line = line.strip()  # string leading/trailing spaces
        line = line.lower()  # clean to all lower case
        line = line.translate(line.maketrans("", "", string.punctuation))  # strip punctuation
        line = ' '.join(line.split())  # strip excess spaces
        text = line.split(' ')  # split each line into a list of words
        for word in text:  # get the counts for each word in the file
            if not word.isnumeric():  # do not allow arabic numerals
                if word in d:  # add to word counter if it exists in the dictionary
                    d[word] = d[word] + 1
                else:  # otherwise add as a new key
                    d[word] = 1
    keys = list(d.keys())
    if flag:  # if we're checking the ham prob
        ham_len = word_counter(ham_dict)  # count the number of words in the ham training dictionary
        prob = np.log(ham_prob)  # start with the natural log of Pr[ham] from the training set
        for key in keys:  # for every value in this file's dictionary
            if key in list(ham_dict.keys()):  # if it exists in the ham dictionary from training
                d[key] = d[key] * np.log(
                    (ham_dict[key] + 1) / (ham_len + m))  # equivalent to slide 25 lecture 8 but using natural log
            else:  # if the word hasn't been seen before
                d[key] = d[key] * np.log(1 / (len(ham_dict) + m))  # same formula, but this time the numerator is (0+1)
            prob = prob + d[key]  # add up the probabilities (ln of a product is the sum of the lns)
    else:  # if we're checking the spam prob
        spam_len = word_counter(spam_dict)  # count the number of words in the spam training dictionary
        prob = np.log(spam_prob)  # start with the natural log of Pr[spam] from the training set
        for key in keys:  # for every value in this file's dictionary
            if key in list(spam_dict.keys()):  # if it exists in the spam dictionary from training
                d[key] = d[key] * np.log(
                    (spam_dict[key] + 1) / (spam_len + m))  # equivalent to slide 25 lecture 8 but using natural log
            else:  # if the word hasn't been seen before
                d[key] = d[key] * np.log(1 / (len(spam_dict) + m))  # same formula, but this time the numerator is (0+1)
            prob = prob + d[key]  # add up the probabilities (ln of a product is the sum of the lns)
    return prob


# function name: prob_finder
# arguments: the number of unique words m, the ham and spam dictionaries from training, the list of folders in the test
# directory, the Pr[y]s as ham/spam_prob
# purpose: builds a dictionary with predictions on if a file is ham or spam
# returns: a prediction dictionary where the keys are the file names
def prob_finder(m, ham_dict, spam_dict, test_directory, ham_prob, spam_prob):
    d = dict()  # init empty dictionary d
    for directory in test_directory:  # in both ham and spam folders in the test folder
        this_dir = listdir(test_set + directory)  # list of files in the current folder
        for file in this_dir:  # iterate through files, find the ham and spam probability for each one
            file_prob_ham = get_file_prob(file, True, directory, ham_dict, spam_dict, ham_prob, spam_prob, m)
            file_prob_spam = get_file_prob(file, False, directory, ham_dict, spam_dict, ham_prob, spam_prob, m)
            if file_prob_ham >= file_prob_spam:  # if the ham prob is greater than or equal to the spam prob
                d[file] = 'ham'  # assign this file as ham
            else:  # otherwise assign as spam
                d[file] = 'spam'
    return d  # return your prediction dictionary


# function name: get_accuracy
# arguments: a prediction dictionary
# purpose: finds the accuracy of the prediction dictionary
# returns: returns accuracy as a float
def get_accuracy(dictionary):
    denominator = len(dictionary)  # number of files
    numerator = 0  # init as 0
    for key in list(dictionary.keys()):  # iterate through each key (file name)
        if dictionary[key] in key:  # if the prediction is in the file name (every file has ham or spam in the title)
            numerator = numerator + 1  # add to the number of correctly classified files
    accuracy = numerator / denominator  # correctly classified files/total number of files
    return accuracy


main()  # main call
