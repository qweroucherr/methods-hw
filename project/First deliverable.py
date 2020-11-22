import numpy as np
import pandas as pd
import jieba
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

def make_label(star):
    if star > 3:
        return 1
    else:
        return 0
    
def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def evaluate(data, model):
    with torch.no_grad():
        y_predicted = model(data.feature)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
        return acc

def weight_reset(m):
    # I grab this func from website. It is called to compare optimizers.
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

        
if __name__ == "__main__":
    
    
    stop_words_file = '哈工大停用词表.txt'
    stopwords = get_custom_stopwords(stop_words_file)

    vect = CountVectorizer(max_df = 0.8, 
                           min_df = 0.1, 
                           max_features = 10000,
                           token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', 
                           stop_words=frozenset(stopwords))

    # data preparation
    # data = pd.read_csv('data0.csv', nrows=20000)
    data = pd.read_csv('data0.csv')    
    data['sentiment'] = data.star.apply(make_label)
    data['cut_comment'] = data.comment.apply(chinese_word_cut)

    X = data['cut_comment']
    y = data.sentiment

    X = pd.DataFrame(vect.fit_transform(X).toarray(), columns=vect.get_feature_names())

    device = torch.device("cuda")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    X_train = torch.tensor(X_train.values, dtype=torch.float, device = device)
    X_test = torch.tensor(X_test.values, dtype=torch.float, device = device)
    y_train = torch.tensor(y_train.values, dtype=torch.float, device = device)
    y_train = y_train.view(y_train.shape[0], 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float, device = device)
    y_test = y_test.view(y_test.shape[0], 1)

    # define model


    n_samples, n_features = X_train.shape

    model = Model(n_features)
    model.to(device)
    num_epochs = 100
    learning_rate = 0.1
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # iteration
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    # test
    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f'accuracy: {acc.item():.4f}')