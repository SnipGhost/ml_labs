def generate_net(input_shape, num_classes = 10):
    # --------------------------------------------------------------------------------------------------
    # Modificated VGG
    #
    # X -> [CONV->POOL]x3 -> [DENSE]x2 -> DENSE -> Y
    # --------------------------------------------------------------------------------------------------
    from keras.models import Sequential
    from keras.layers import Input, Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    # --------------------------------------------------------------------------------------------------
    # Осноные настройки сети
    kernel_size = (5, 5)
    pool_size   = (2, 2)
    blocks_size = [ 64, 128, 256]
    blocks_drop = [0.2, 0.2, 0.2]
    denses_size = [512, 256]
    denses_drop = [0.5, 0.5]
    # --------------------------------------------------------------------------------------------------
    # Формируем сеть
    # --------------------------------------------------------------------------------------------------
    model = Sequential()

    for idx, block_size in enumerate(blocks_size):
        if (idx == 0):
            model.add(Conv2D(block_size, kernel_size, activation='relu', input_shape=input_shape))
        else:
            model.add(Conv2D(block_size, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(blocks_drop[idx]))

    # Переформирует Nx28x28x1 в Nx784, чтобы подавать на вход полносвязному слою
    model.add(Flatten())

    for idx, dense_size in enumerate(denses_size):
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dropout(denses_drop[idx]))

    model.add(Dense(num_classes, activation='softmax')) # Выходной слой
    # --------------------------------------------------------------------------------------------------
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # --------------------------------------------------------------------------------------------------
    return model, 'vgg' # второй параметр - имя сети, для сохранения моделей