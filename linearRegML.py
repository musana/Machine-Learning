import math, random as r
from matplotlib import pyplot as plt

x = list(range(-5,6))

def f(x):
    return list(map(lambda q: 1/(1+math.pow(math.e, -q)), x))

y = f(x)
plt.scatter(x,y)
#plt.show()

noktalar = [(r.randint(1,15), r.randint(1,15)) for i in range(15)]

def d(rPoint, point):
    return math.sqrt(math.pow(rPoint[0]-point[0], 2) + math.pow(rPoint[1]-point[1],2))

mesafeler = list(map(lambda w: d((0,0), w), noktalar))

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn import neighbors

df = read_csv("ogrenciPuan.csv")
fea = ["Hours", "Hours2"]

x = df[fea]
y = df["Scores"]


#lg = LinearRegression()
lg = neighbors.KNeighborsClassifier(n_neighbors=2)
x_egitim, x_test, y_egitim, y_test = train_test_split(x,y,test_size=0.2)
lg.fit(x_egitim, y_egitim)
print(y_test)
print(lg.predict(x_test))

print(lg.score(x_egitim, y_egitim))
print(lg.score(x_test, y_test))
