import tensorflow as tf
import numpy as np
from tensorflow import * # or "from tensorflow import keras"

# 1 layer, 1 neuron, 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# initialise input
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# train model
model.fit(xs, ys, epochs=500)

# test model with a specific value (in this case 10)
print(model.predict([10.0]))