from pyspark import SparkContext, SparkConf
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston
from datetime import date
import re

sc = SparkContext(appName = 'rf')
path = '/Users/nali/Beifei/elephant_teeth/Rossman/code20151028/'

p_quote = re.compile('"(\d+)"')
def splitx(x):
    x_sep = x.split(',')
    x_sep2 = map(lambda xx: int(xx), x_sep[2].split('-'))
    date0 = date(x_sep2[0], x_sep2[1], x_sep2[2])
    date1 = date(2013, 01, 01)
    days = (date0 - date1).days
    stateholiday_num = {'0':0, 'a':1, 'b':2, 'c':3}
    stateholiday = p_quote.findall(x_sep[7])[0]
    state_num = stateholiday_num[stateholiday]
    schoolholiday = int(p_quote.findall(x_sep[8])[0])
    x_sep.remove(x_sep[2])
    x_sep.remove(x_sep[6])
    x_sep.remove(x_sep[6])
    x_int = map(lambda xx: int(xx), x_sep)
    features = [x_int[0], x_int[1], days, x_int[4], x_int[5], state_num, schoolholiday]
    sales = x_int[2]
    customers = x_int[3]
    return (features, sales) 


traindata = sc.textFile(path+'train.csv')\
        .map(lambda x: splitx(x))
train_x = np.array(traindata.map(lambda x: x[0]).collect())
train_y = np.array(traindata.map(lambda x: x[1]).collect())

testdata = sc.textFile(path+'test.csv')\
        .map(lambda x: splitx(x))
test_x = np.array(testdata.map(lambda x: x[0]).collect())
test_y = np.array(testdata.map(lambda x: x[1]).collect())

size = len(train_x)
rf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=1)
rf.fit(train_x, train_y)

def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up

err_down, err_up = pred_ints(rf, test_x, percentile=90)
truth = test_y
correct = 0.
for i, val in enumerate(truth):
    if err_down[i] <= val <= err_up[i]:
        correct += 1
print correct/len(truth)
