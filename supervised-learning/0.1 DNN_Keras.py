# compare algorithms
from pandas import read_csv
import matplotlib as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing, metrics
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#TRANSFORM LABEL TO INTEGER
# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()
# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
Y_label = le.fit_transform(y)

Y_label = np.reshape(Y_label,(-1, 1))
enc = preprocessing.OneHotEncoder()
# 2. FIT
enc.fit(Y_label)
# 3. Transform
onehotlabels = enc.transform(Y_label).toarray()

X=normalizedX
y=onehotlabels

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

#Create neural net
model = Sequential()
model.add(Dense(10, input_dim=normalizedX.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(50, input_dim=normalizedX.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(10, input_dim=normalizedX.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(onehotlabels.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','Precision'])
history = model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation), verbose=1, epochs=50)

# Measure accuracy
prediction = model.predict(X_validation)
y_score=prediction
pred = np.argmax(prediction,axis=1)
y_eval = np.argmax(Y_validation,axis=1)
score = metrics.accuracy_score(y_eval, pred)
print("Validation score: {}".format(score))

#Confusion Matrix
print("MATRIX")
cfm = confusion_matrix(y_eval,pred)
print(cfm)

#Report
cmp = classification_report(y_eval,pred)
print(cmp)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()