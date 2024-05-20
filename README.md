# Python : Keras (hight-level neural networks API) with dataset MNIST

&nbsp;

**Deep learning for humans.**<br />
Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. Keras also gives the highest priority to crafting great documentation and developer guides.

&nbsp;

Reference : <br />
- Documentations | Keras
  <pre>https://keras.io/</pre>

&nbsp;

&nbsp;

---

&nbsp;

## Project

Dalam project ini, kita akan membuat dan melatih sebuah model jaringan saraf tiruan sederhana untuk melakukan klasifikasi gambar menggunakan dataset MNIST, yang terdiri dari gambar-gambar angka tulisan tangan.

&nbsp;

&nbsp;

## Begin : 

Creating Directories and File Structures
<pre>
  ❯ mkdir -p data models utils

  ❯ touch data/__init__.py data/data_loader.py

  ❯ touch models/__init__.py models/neural_network.py

  ❯ touch utils/__init__.py utils/helper_functions.py

  ❯ touch train.py predict.py

  ❯ tree -L 2 -a -I 'README.md|.DS_Store|.git|.gitignore|venv' ./
    ./
    ├── data
    │   ├── __init__.py
    │   └── data_loader.py
    ├── models
    │   ├── __init__.py
    │   └── neural_network.py
    ├── predict.py
    ├── train.py
    └── utils
        ├── __init__.py
        └── helper_functions.py

    3 directories, 8 files
</pre>

&nbsp;

&nbsp;

&nbsp;

## Codes : 

<pre>
  ❯ vim data/data_loader.py
</pre>
<pre>
  import numpy as np
  from keras.datasets import mnist
  from keras.utils import to_categorical

  def load_mnist_data():
      (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
      
      train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
      test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
      
      train_labels = to_categorical(train_labels)
      test_labels = to_categorical(test_labels)
      
      return (train_images, train_labels), (test_images, test_labels)
</pre>

&nbsp;

<pre>
  ❯ vim models/neural_network.py
</pre>
<pre>
  from keras.models import Sequential
  from keras.layers import Dense

  def create_neural_network():
      model = Sequential()
      model.add(Dense(64, activation='relu', input_shape=(28 * 28,)))
      model.add(Dense(64, activation='relu'))
      model.add(Dense(10, activation='softmax'))
      
      return model
</pre>

&nbsp;

<pre>
  ❯ vim utils/helper_functions.py
</pre>
<pre>
import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
</pre>

&nbsp;

<pre>
  ❯ vim train.py
</pre>
<pre>
  from data.data_loader import load_mnist_data
  from models.neural_network import create_neural_network
  from utils.helper_functions import plot_training_history

  (train_images, train_labels), (test_images, test_labels) = load_mnist_data()

  model = create_neural_network()
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

  plot_training_history(history)

  model.save('mnist_model.h5')
</pre>

&nbsp;

<pre>
  ❯ vim predict.py
</pre>
<pre>
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
</pre>

&nbsp;

&nbsp;

---

&nbsp;