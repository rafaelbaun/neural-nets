import keras as keras
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

train_df = pd.read_csv('./data/train.csv')

#Daten müssen gemischt werden, da Daten nahe beieinander liegen können
np.random.shuffle(train_df.values)

print(train_df.head())

model = keras.Sequential([
    keras.layers.Dense(512,input_shape=(2,),activation = 'relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(256,activation = 'relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(256,activation = 'relu'),
    keras.layers.Dense(2, activation = 'sigmoid')])

model.compile(optimizer = 'adam',
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

#Für den X des Mdels müssen die Daten der X spalte und Y Spalte gestacked werden und als np array mitgegeben werden
x = np.column_stack((train_df.x.values,train_df.y.values))
model.fit(x,train_df.color.values, batch_size = 32, epochs = 20 )

#testen wie das Model arbeitet
test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values,test_df.y.values))

print("EVALUATION")
model.evaluate(test_x,test_df.color.values)