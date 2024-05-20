import numpy as np
from keras.models import load_model
from data.data_loader import load_mnist_data

(train_images, train_labels), (test_images, test_labels) = load_mnist_data()

model = load_model('mnist_model.h5')

# Prediksi label untuk beberapa gambar uji
predictions = model.predict(test_images[:5])

# Tampilkan prediksi dan label sebenarnya
for i, prediction in enumerate(predictions):
    predicted_label = np.argmax(prediction)
    actual_label = np.argmax(test_labels[i])
    print(f'Predicted: {predicted_label}, Actual: {actual_label}')
