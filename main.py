from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime 

print("Hello")

data=pd.read_csv("MicrosoftStock.csv")
# print(data.head())

# print(data.info())
# print(data.describe())

plt.figure(figsize=(12,6))
# open and close prices
plt.plot(data["date"],data["open"],label="Open",color="blue")
plt.plot(data["date"],data["close"],label="Close",color="red")
plt.title("Open-Close Price over Time")
plt.legend()
plt.show() 

# Trading Volume (checking for outliers)
plt.figure(figsize=(12,6))
plt.plot(data["date"],data["volume"],label="Volume",color="orange")
plt.title("Stock Volume over Time")
plt.show()

# droping none numeric data 

numeric_data=data.select_dtypes(include=["int64","float64"])

# checkin coorelation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(),annot=True,cmap="coolwarm")
plt.title("Feature_coorelation")
plt.show()

data["date"]=pd.to_datetime(data["date"])

prediction=data.loc[
    (data["date"]> datetime(2013,1,1))&
    (data["date"]< datetime(2018,1,1))
]


plt.figure(figsize=(12,6))
plt.plot(data["date"],data["close"],color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")
plt.show()

stock_close=data.filter(["close"])
dataset=stock_close.values #convert to numpy array
training_data_len=int(np.ceil(len(dataset))* 0.95)

scaler=StandardScaler()
scaled_data=scaler.fit_transform(dataset)

training_data=scaled_data[:training_data_len]

X_train,y_train=[],[]

for i in range(60,len(training_data)):
    X_train.append(training_data[i-60:i,0])
    y_train.append(training_data[i,0])

X_train,y_train=np.array(X_train),np.array(y_train)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

model=keras.models.Sequential()

model.add(keras.layers.LSTM(64,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(keras.layers.LSTM(64,return_sequences=False))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",
              loss="mae",
              metrics=[keras.metrics.RootMeanSquaredError()])

training=model.fit(X_train,y_train,epochs=20,batch_size=32)
test_data=scaled_data[training_data_len-60:]
X_test,y_test=[],dataset[training_data_len:]

for i in range(60,len(test_data)):
    X_test.append(test_data[i-60:i,0])

X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predictions=model.predict(X_test)
predictions=scaler.inverse_transform(predictions)

train=data[:training_data_len]
test=data[training_data_len:]
test=test.copy()
test["Predictions"]=predictions

plt.figure(figsize=(12,8))
plt.plot(train["date"],train["close"],label="Train (Actual)",color="blue")
plt.plot(test["date"],test["close"],label="Test (Actual)",color="orange")
plt.plot(test["date"],test["Predictions"],label="Predictions",color="red")
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()