# 1. SORU
import math
import matplotlib.pyplot as plt
import random as r

x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
y = []

def f():
    for i in x:
        y.append(1/1+math.pow(math.e, -i))
    plt.plot(x,y)
    plt.show() 
f()

# 2. SORU
noktalar  = []
mesafeler = []

def d(x1, x2):
    return math.sqrt(math.pow(x1[0]-x2[0],2)+math.pow(x1[1]-x2[1],2))

for i in range(15):
    noktalar.append((r.randint(1,15), r.randint(1,15)))

for i in noktalar:
    mesafeler.append(d((0,0), i))

# 3. SORU
df = read_csv("ogrenciPuan.txt")
plt.plot(df["saat"], df["yuzdelik_puan"])
plt.show()

lg = LineerRegression()
x_egitim, x_test, y_egitim, y_test = lg.train_test_split(df["saat"], df["yuzdelik_puan"], test_size=0.2)
q = lg.fit(x_egitim, y_egitim)
lg.score(x_egitim, y_egitim)
lg.score(x_test, y_test)
