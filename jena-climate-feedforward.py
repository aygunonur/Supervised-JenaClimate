import numpy as np

PREDICT_INTERVAL = 144

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

training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x[:-PREDICT_INTERVAL], dataset_y[PREDICT_INTERVAL:], test_size=0.2)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(training_dataset_x)
training_dataset_x = ss.transform(training_dataset_x)
test_dataset_x = ss.transform(test_dataset_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential(name='Jena-Climate-FeedForward')
model.add(Dense(128, activation='relu', input_dim=training_dataset_x.shape[1], name='Dense-1'))
model.add(Dropout(0.3, name='Dropout-1'))
model.add(Dense(128, activation='relu', name='Dense-2'))
model.add(Dropout(0.3, name='Dropout-2'))
model.add(Dense(1, activation='linear', name='Output'))

model.summary()

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping

esc = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=20, validation_split=0.2, callbacks=[esc])

eval_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]} --> {eval_result[i]}')

predict_data = np.array([996.52, -8.02, 265.40, -8.90, 93.30, 3.33, 3.11, 0.22, 1.94, 3.12, 1307.75, 1.03, 1.75, 152.30], dtype=np.float32)

predict_data = ss.transform(predict_data.reshape(1, -1))

predict_result = model.predict(predict_data)
print(predict_result[0][0])