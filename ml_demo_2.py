import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), # 128 can change to 1024
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(256, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


model.compile(optimizer = tf.keras.optimizers.Adam(), # or optimizer='adam'
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy']) # can also remove metrics=['accuracy']

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])