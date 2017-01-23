from keras.models import Sequential
from keras.layers import Dense
from termcolor import cprint
import numpy

# datasetimizi yüklüyoruz.
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Dateseti ayrıştır. X, 8 adet girdimiz. Y, ise çıkışımız.
X = dataset[:600,0:8]
Y = dataset[:600,8]

# Modelimizi oluşturuyoruz.
model = Sequential()
model.add(Dense(30, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(20, init='uniform', activation='relu'))
model.add(Dense(12, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Modelimizi derliyoruz.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modelimizi eğitiyoruz.
model.fit(X, Y, nb_epoch=150, batch_size=10,  verbose=1)


# Modelimizin başarı yüzdesini hesaplıyoruz.
scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("-"*150)

# Test verimizi veriyoruz.
test_verisi = dataset[600:696, 0:8]
predictions = model.predict(test_verisi)

dogru = 0
yanlis = 0
toplam_veri = len(dataset[600:696,8])

for x, y in zip(predictions, dataset[600:696,8]):    
    x = int(numpy.round(x[0]))
    if int(x) == y:
        cprint("Tahmin: "+str(x)+" - Gerçek Değer: "+str(int(y)), "white", "on_green", attrs=['bold'])
        dogru += 1
    else:
        cprint("Tahmin: "+str(x)+" - Gerçek Değer: "+str(int(y)), "white", "on_red", attrs=['bold'])
        yanlis += 1

print("\n", "-"*150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru, "\nYanlış Bilme Sayısı: ",yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100*dogru/toplam_veri))+"%", sep="")
