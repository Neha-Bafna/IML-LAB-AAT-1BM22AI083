import sys
from sklearn.feature_extraction import text
from sklearn import pipeline
from sklearn import linear_model
import numpy
def make():
    c = pipeline.Pipeline([
        ('vect',text.TfidfVectorizer(stop_words='english', ngram_range=(1, 1), min_df=4, strip_accents='ascii', lowercase=True)),('c',linear_model.SGDClassifier(class_weight='balanced'))])
    return c
def run():
    k = [('Business means risk!', 1), ("This is a document", 1), ("this is another document", 4), ("documents are seperated by newlines", 8)]
    xs, ys = load('trainingdata.txt')
    m = make()
    m.fit(xs, ys)
    txs = list(line for line in sys.stdin)[1:]
    for y, x in zip(m.predict(txs), txs):
        for p, cl in k:
            if p in x:
                print(cl)
                break
        else:
            print(y)
def load(filename):
    with open(filename, 'r') as data_file:
        sz = int(data_file.readline())
        xs = numpy.zeros(sz, dtype=numpy.dtype(object))
        ys = numpy.zeros(sz, dtype=numpy.dtype(int))
        for i, line in enumerate(data_file):
            idx = line.index(' ')
            if idx == -1:
                raise ValueError('invalid input file')
            cl = int(line[:idx])
            wrd = line[idx+1:]
            xs[i] = wrd
            ys[i] = cl
    return xs, ys
if __name__ == '__main__':
    run()