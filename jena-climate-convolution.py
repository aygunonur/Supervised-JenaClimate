import numpy as np

PREDICT_INTERVAL = 144
LOOKBACK_INTERVAL = 144
BATCH_SIZE = 32

dataset_x = np.loadtxt('jena_climate_2009_2016.csv', delimiter=',', skiprows=1, usecols=range(1, 15), dtype=np.float32)

dataset_y = dataset_x[:, 1]

import matplotlib.pyplot as plt

figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Jena Climate Celsius for 20 Days')
plt.xlabel('Days')
plt.ylabel('Celsius')
plt.xticks(range(1, 51))
plt.plot(range(1, 51), dataset_y[0:PREDICT_INTERVAL * 50:PREDICT_INTERVAL])
plt.show()

from sklearn.model_selection import train_test_split

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.2, shuffle=False)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
training_dataset_x = ss.transform(training_dataset_x)
test_dataset_x = ss.transform(test_dataset_x)

training_dataset_x, validation_dataset_x, training_dataset_y, validation_dataset_y = train_test_split(training_dataset_x, training_dataset_y, test_size=0.2, shuffle=False)

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, dataset_x, dataset_y, batch_size):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.batch_size = batch_size
        self.indices = np.arange(0, (len(dataset_x) - PREDICT_INTERVAL) // self.batch_size, dtype=np.int32)
    
    def __len__(self):
        return (len(self.dataset_x) - PREDICT_INTERVAL) // self.batch_size
    
    def __getitem__(self, index):
        result_x = np.zeros((self.batch_size, LOOKBACK_INTERVAL, self.dataset_x.shape[1]))
        result_y = np.zeros(self.batch_size)
        
        for i in range(self.batch_size):
            start = self.batch_size * self.indices[index] + i
            result_x[i] = self.dataset_x[start:start + LOOKBACK_INTERVAL]
            result_y[i] = self.dataset_y[start + PREDICT_INTERVAL]
         
        return result_x, result_y
       
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout

model = Sequential(name='Jena-Climate-Convolution')
model.add(Conv1D(32, kernel_size=3, strides=3, input_shape=(LOOKBACK_INTERVAL, training_dataset_x.shape[1]), name='Convolution'))
model.add(Flatten(name='flatten'))
model.add(Dense(64, activation='relu', name='Dense-1'))
model.add(Dropout(0.3, name='Dropout-1'))
model.add(Dense(64, activation='relu', name='Dense-2'))
model.add(Dropout(0.3, name='Dropout-2'))
model.add(Dense(1, activation='linear', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
hist = model.fit(DataGenerator(training_dataset_x, training_dataset_y, BATCH_SIZE), validation_data = DataGenerator(validation_dataset_x, validation_dataset_y, BATCH_SIZE), epochs=3, callbacks=[esc])

eval_result = model.evaluate(DataGenerator(test_dataset_x, test_dataset_y, BATCH_SIZE))
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]} --> {eval_result[i]}')

predict_data = dataset_x[5555:5555 + PREDICT_INTERVAL]
predict_data = ss.transform(predict_data)
predict_result = model.predict(np.expand_dims(predict_data, axis=0))
print(f'Predicted result: {predict_result[0]}')
print(f'Real result: {dataset_x[5555 + PREDICT_INTERVAL][1]}')
