from keras.models import Sequential
from keras.layers import Dense

def create_neural_network():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model
