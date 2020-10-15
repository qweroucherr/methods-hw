# Alex Jian Zheng
import random
import numpy as np
from math import exp, log
from collections import defaultdict

import argparse

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Note: Prevents overflow of exp by capping activation at 20.
    The `np.sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``
    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * np.sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = np.zeros(len(vocab))
        self.df = np.array(df)
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                # Creating feature vectors based on term frequency
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1
    
    def tfidf(self,tnd):
        '''
        using tf-idf to replace term frequency as the features
        :param tnd: The total number of documents
        '''
        tf = self.x/sum(self.x)
        self.df[0]=tnd
        idf = np.log(tnd/self.df)
        self.x = tf*idf
        self.x[0] = 1
    

class LogReg:
    def __init__(self, num_features, learning_rate=0.05, idf=False,tnd=1):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param learning_rate: How big of a SG step we take
        :param idf: Using tf-id to replace term frequency as the features 
        :param tnd: The total number of documents
        """

        self.beta = np.zeros(num_features)
        self.learning_rate = learning_rate
        self.idf = idf
        self.tnd = tnd

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            if self.idf == True:
                ii.tfidf(self.tnd)
            p = sigmoid(np.dot(self.beta,ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, regularization = 0):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param regularization: default 0
        :return: The current vector of parameters
        """

        # Your code here
        if self.idf == True:
            train_example.tfidf(self.tnd)
        gradient = np.zeros(len(self.beta))
        gradient = train_example.x * (train_example.y - sigmoid(np.dot(self.beta,train_example.x)))
        self.beta += (gradient-self.beta*regularization*2) * self.learning_rate
        return self.beta

    def update_learning_rate(self,step,iteration):
        self.learning_rate = pow(step,iteration+1)


def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples

    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """

    # You should not need to modify this function
    
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab, df

def dict_sort(dict_input):
    '''
    A function to sort the final weights
    '''
    aux = [(dict_input[key],key) for key in dict_input]
    aux.sort()
    aux.reverse()
    result = [(w[1],w[0]) for w in aux]
    result = dict(result)
    return result


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    #''' for switch between the toy and real examples
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="vocab", required=False)
    #'''
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)
    argparser.add_argument("--regularization", help="Regularization parameter",
                           type=float, default=0, required=False)
    argparser.add_argument("--scheduled", help="Scheduled update of learning rate",
                           type=bool, default=False, required=False)
    argparser.add_argument("--tfidf", help="Using tfidf as features",
                           type=bool, default=False, required=False)
    ''' for switch between the toy and real examples
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="toy_positive.txt", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="toy_negative.txt", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="toy_vocab.txt", required=False)
    '''

    args = argparser.parse_args()
    train, test, vocab, df = read_dataset(args.positive, args.negative, args.vocab)
    tnd = len(train)+len(test)
    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.step,idf=args.tfidf,tnd=tnd)
    # Iterations
    update_number = 0
    performance = np.zeros((len(train)*args.passes//5+1,4))
    for pp in range(args.passes):
        if args.scheduled == True:
            lr.update_learning_rate(args.step, pp)
        for ii in train:
            lr.sg_update(ii, regularization = args.regularization)
            
            update_number += 1
            
            if update_number % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                #''' To store the performance for plots in the analysis
                performance[update_number//5]=[train_lp,ho_lp,train_acc,ho_acc]
                #'''
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))
    #The following code is for determining the best and worst predictors       
    '''
    features = dict(zip(vocab, lr.beta))
    features_sorted = dict_sort(features)
    print("Top 10 features")
    print(list(features_sorted.keys())[:10])
    print("Bottom 10 features ")
    print(list(features_sorted.keys())[-10:])
    '''