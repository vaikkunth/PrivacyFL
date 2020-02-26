import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
digits = load_digits()
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.80, random_state=0)

X = digits.data
y = digits.target

print(len(X))

X = np.loadtxt("adult_clean.csv", usecols=(0, 2, 4, 10, 11, 12), skiprows=1, delimiter=',')
# Outcome variable

Y = np.loadtxt("adult_clean.csv", usecols=14, dtype=str, skiprows=1, delimiter=',')


'''

# 3 clients, 500 each
X = digits.data
Y = digits.target
X_train1, y_train1 = X[:500], Y[:500]
X_train2, y_train2 = X[500:1000], Y[500:1000]
X_train3, y_train3 = X[1000:1500], Y[1000:1500]

X_test, y_test = X[1500:], Y[1500:]

logisticRegr1 = LogisticRegression()
logisticRegr1.fit(X_train1, y_train1)
score = logisticRegr1.score(X_test, y_test)
print(score)

logisticRegr2 = LogisticRegression()
logisticRegr2.fit(X_train2, y_train2)
score = logisticRegr2.score(X_test, y_test)
print(score)

logisticRegr3 = LogisticRegression()
logisticRegr3.fit(X_train3, y_train3)
score = logisticRegr3.score(X_test, y_test)
print(score)

coef = np.average([logisticRegr1.coef_, logisticRegr2.coef_, logisticRegr3.coef_], axis = 0)

logisticRegr3.coef_ = coef
score = logisticRegr3.score(X_test, y_test)
print(score)
