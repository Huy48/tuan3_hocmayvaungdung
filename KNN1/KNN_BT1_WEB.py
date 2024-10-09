import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
from flask import Flask, Response
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import io

matplotlib.use("SVG")
app = Flask(__name__)

def loadCsv(filename) -> pd.DataFrame:
    return pd.read_csv(filename)

def splitTrainTest(data, ratio_test):
    np.random.seed(28)
    index_permu = np.random.permutation(len(data))
    data_permu = data.iloc[index_permu]
    len_test = int(len(data_permu) * ratio_test)
    test_set = data_permu.iloc[:len_test, :]
    train_set = data_permu.iloc[len_test:, :]
    X_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]
    return X_train, y_train, X_test, y_test

def get_words_frequency(data_X):
    bag_words = np.concatenate([i[0].split(' ') for i in data_X.values], axis=None)
    bag_words = np.unique(bag_words)
    matrix_freq = np.zeros((len(data_X), len(bag_words)), dtype=int)
    word_freq = pd.DataFrame(matrix_freq, columns=bag_words)
    for id, text in enumerate(data_X.values.reshape(-1)):
        for j in bag_words:
            word_freq.at[id, j] = text.split(' ').count(j)
    return word_freq, bag_words

def transform(data_test, bags):
    matrix_0 = np.zeros((len(data_test), len(bags)), dtype=int)
    frame_0 = pd.DataFrame(matrix_0, columns=bags)
    for id, text in enumerate(data_test.values.reshape(-1)):
        for j in bags:
            frame_0.at[id, j] = text.split(' ').count(j)
    return frame_0

def cosine_distance(train_X_number_arr, test_X_number_arr):
    dict_kq = dict()
    for id, arr_test in enumerate(test_X_number_arr, start=1):
        q_i = np.sqrt(sum(arr_test**2))
        for j in train_X_number_arr:
            _tu = sum(j * arr_test)
            d_j = np.sqrt(sum(j**2))
            _mau = d_j * q_i
            kq = _tu / _mau
            if id in dict_kq:
                dict_kq[id].append(kq)
            else:
                dict_kq[id] = [kq]
    return dict_kq

class KNNText:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.X_test = X_test
        _distance = cosine_distance(self.X_train, self.X_test)
        self.y_train.index = range(len(self.y_train))
        _distance_frame = pd.concat([pd.DataFrame(_distance), pd.DataFrame(self.y_train)], axis=1)
        target_predict = dict()
        for i in range(1, len(self.X_test) + 1):
            subframe = _distance_frame[[i, 'Label']].sort_values(by=i).head(self.k)
            most_frequent = subframe['Label'].value_counts().idxmax()
            target_predict[i] = [most_frequent]
        return target_predict

@app.route('/')
def home():
    return '<h1>Welcome to the KNN Text Classification App!</h1><p>Go to <a href="/print-plot">/print-plot</a> to see the confusion matrix plot.</p>'

@app.route('/print-plot')
def plot_png():
    data = loadCsv('C:\\Users\\My Laptop\\Desktop\\học máy và ứng dụng(TH)\\lab3\\Education.csv')
    data['Text'] = data['Text'].apply(lambda x: x.replace(',', ''))
    data['Text'] = data['Text'].apply(lambda x: x.replace('.', ''))
    X_train, y_train, X_test, y_test = splitTrainTest(data, 0.25)
    words_train_fre, bags = get_words_frequency(X_train)
    words_test_fre = transform(X_test, bags)
    knn = KNNText(k=2)
    knn.fit(words_train_fre.values, y_train)
    pred_ = pd.DataFrame(pd.DataFrame(knn.predict(words_test_fre.values)).values.reshape(-1), columns=['Predict'])
    pred_.index = range(1, len(pred_) + 1)
    y_test.index = range(1, len(y_test) + 1)
    y_test = y_test.to_frame(name='Actual')
    cm = confusion_matrix(y_test, pred_, labels=["positive", "negative"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=["positive", "negative"],
                yticklabels=["positive", "negative"], ax=ax)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    plt.tight_layout()

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)

    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
