import keras as keras
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

train_df = pd.read_csv('./data/train.csv')
print(train_df.head())
#Da Da die farben Strings sind, müssen diese mithilfe eines Dictionaries gemappt werden
color_dict = {'red': 0, 'blue': 1, 'green': 2, 'teal': 3, 'orange': 4, 'purple': 5}
train_df['color'] = train_df.color.apply(lambda x: color_dict[x])
print(train_df.head())
print(train_df.color.unique())
#Daten müssen gemischt werden, da Daten nahe beieinander liegen können
np.random.shuffle(train_df.values)




model = keras.Sequential([
    keras.layers.Dense(32,input_shape=(2,),activation = 'relu'),
    keras.layers.Dense(32,activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(6, activation = 'sigmoid')])

model.compile(optimizer = 'adam',
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

#Für den X des Mdels müssen die Daten der X spalte und Y Spalte gestacked werden und als np array mitgegeben werden
x = np.column_stack((train_df.x.values,train_df.y.values))
model.fit(x,train_df.color.values, batch_size = 4, epochs = 10 )

#testen wie das Model arbeitet
test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values,test_df.y.values))
test_df['color'] = test_df.color.apply(lambda x: color_dict[x])

print("EVALUATION")
model.evaluate(test_x,test_df.color.values)

print("Prediction",np.round(model.predict(np.array([[0,3]]))))