from data.data_loader import load_mnist_data
from models.neural_network import create_neural_network
from utils.helper_functions import plot_training_history

(train_images, train_labels), (test_images, test_labels) = load_mnist_data()

model = create_neural_network()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

plot_training_history(history)

model.save('mnist_model.h5')
