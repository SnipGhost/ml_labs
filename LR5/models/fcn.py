def generate_net(input_shape, num_classes = 10):
    # --------------------------------------------------------------------------------------------------
    # Classic full connected NN
    #
    # X -> DENSE -> DENSE -> DENSE -> Y
    # --------------------------------------------------------------------------------------------------
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    # --------------------------------------------------------------------------------------------------
    model = Sequential()
    model.add(Dense(units=600, activation='relu', input_dim=len(pixels)))
    model.add(Dense(units=80, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    # --------------------------------------------------------------------------------------------------
    sgd = optimizers.SGD(lr=0.08, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # --------------------------------------------------------------------------------------------------
    return model, 'fcn' # Второй параметр - имя сети, для сохранения моделей