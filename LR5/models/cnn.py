def generate_net(input_shape, num_classes = 10):
    # --------------------------------------------------------------------------------------------------
    # Classic convolution model
    #
    # X -> CONV -> POOL -> CONV -> POOL -> DENSE -> DENSE -> Y
    # --------------------------------------------------------------------------------------------------
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    # --------------------------------------------------------------------------------------------------
    model = Sequential()
    model.add(Conv2D(75, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(100, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # --------------------------------------------------------------------------------------------------
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # --------------------------------------------------------------------------------------------------
    return model, 'cnn' # Второй параметр - имя сети, для сохранения моделей