from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, concatenate
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Helper: Early stopping
early_stopper = EarlyStopping(patience=5)

# set default params
BATCH_SIZE = 256
NB_CLASSES = 228
EPOCHS = 60
PERMISSION_SIZE = 30
API_SIZE = 32
DENSE_UNITS = 1024

def get_dataset(path):
    '''
        Retrieve our dataset from the given path
    '''
    # train data
    train_df = pd.read_csv(path + '/train-9.csv', header=None, skiprows=1)
    train_y = np.array(train_df.iloc[:, 2])
    train_per = np.array(train_df.iloc[:, 3:880])
    train_per = np.concatenate((train_per, np.zeros((train_per.shape[0], 23))), 1)

    train_api = np.array(train_df.iloc[:, 880:])
    train_api = np.concatenate((train_api, np.zeros((train_api.shape[0], 67))), 1)

    unique, counts = np.unique(train_y, return_counts=True)

    train_per = train_per.reshape(train_per.shape[0], PERMISSION_SIZE, PERMISSION_SIZE, 1)
    train_api = train_api.reshape(train_api.shape[0], API_SIZE, API_SIZE, 1)
    train_per = train_per.astype('float32')
    train_api = train_api.astype('float32')
    train_per /= 255
    train_api /= 255

    # validation data
    val_df = pd.read_csv(path + '/file-9.csv', header=None, skiprows=1)
    val_y = np.array(val_df.iloc[:, 2])
    val_per = np.array(val_df.iloc[:, 3:880])
    val_per = np.concatenate((val_per, np.zeros((val_per.shape[0], 23))), 1)

    val_api = np.array(val_df.iloc[:, 880:])
    val_api = np.concatenate((val_api, np.zeros((val_api.shape[0], 67))), 1)

    unique_val, counts_val = np.unique(val_y, return_counts=True)

    val_per = val_per.reshape(val_per.shape[0], PERMISSION_SIZE, PERMISSION_SIZE, 1)
    val_api = val_api.reshape(val_api.shape[0], API_SIZE, API_SIZE, 1)
    val_per = val_per.astype('float32')
    val_api = val_api.astype('float32')
    val_per /= 255
    val_api /= 255

    # test data
    test_df = pd.read_csv(path + '/file-0.csv', header=None, skiprows=1)
    test_y = np.array(test_df.iloc[:, 2])
    test_per = np.array(test_df.iloc[:, 3:880])
    test_per = np.concatenate((test_per, np.zeros((test_per.shape[0], 23))), 1)

    test_api = np.array(test_df.iloc[:, 880:])
    test_api = np.concatenate((test_api, np.zeros((test_api.shape[0], 67))), 1)

    unique_test, counts_test = np.unique(test_y, return_counts=True)

    test_per = test_per.reshape(test_per.shape[0], PERMISSION_SIZE, PERMISSION_SIZE, 1)
    test_api = test_api.reshape(test_api.shape[0], API_SIZE, API_SIZE, 1)
    test_per = test_per.astype('float32')
    test_api = test_api.astype('float32')
    test_per /= 255
    test_api /= 255

    original_test_y = test_y

    train_y = to_categorical(train_y, NB_CLASSES)
    val_y = to_categorical(val_y, NB_CLASSES)
    test_y = to_categorical(test_y, NB_CLASSES)

    train_x = [train_per, train_api]
    test_x = [test_per, test_api]
    val_x = [val_per, val_api]

    return (train_x, train_y, test_x, test_y, val_x, val_y, original_test_y)

def compile_model(network):
    '''
    Compile a sequential model
    Args:
        network (dict): the parameters of the network
    Return:
        a compile network
    '''
    # Get our network parameters
    nb_layers = network['nb_layers']
    # nb_neurons = network['nb_neurons']
    # activation = network['activation']
    optimizer = network['optimizer']

    # per_input = Input(shape=(PERMISSION_SIZE, PERMISSION_SIZE, 1))
    # per_conv = Conv2D(nb_neurons, (2, 2), padding='same', activation=activation, input_shape=(PERMISSION_SIZE, PERMISSION_SIZE, 1))(per_input)
    # per_pool = MaxPooling2D((2, 2), strides=2)(per_conv)
    # for i in range(2, nb_layers):
    #     i //= 2
    #     per_conv = Conv2D(nb_neurons * (2 ** i), (2, 2), padding='same', activation=activation)(per_pool)
    #     per_pool = MaxPooling2D((2, 2), strides=2)(per_conv)
  
    # per_flatten = Flatten()(per_pool)
    # per_dense = Dense(nb_neurons * nb_layers, activation='relu')(per_flatten)
    # per_output = Dense(NB_CLASSES, activation='softmax')(per_dense)
    # per_model = Model(inputs=per_input, outputs=per_output)

    # api_input = Input(shape=(API_SIZE, API_SIZE, 1))
    # api_conv = Conv2D(nb_neurons, (2, 2), padding='same', activation=activation, input_shape=(API_SIZE, API_SIZE, 1))(api_input)
    # api_pool = MaxPooling2D((2, 2), strides=2)(api_conv)
    # for i in range(2, nb_layers):
    #     i //= 2
    #     api_conv = Conv2D(nb_neurons * (2 ** i), (2, 2), padding='same', activation=activation)(api_pool)
    #     api_pool = MaxPooling2D((2, 2), strides=2)(api_conv)
  
    # api_flatten = Flatten()(api_pool)
    # api_dense = Dense(nb_neurons * nb_layers, activation=activation)(api_flatten)
    # api_output = Dense(NB_CLASSES, activation='softmax')(api_dense)
    # api_model = Model(inputs=api_input, outputs=api_output)

    # combined = concatenate([per_model.output, api_model.output])

    # combined_output = Dense(NB_CLASSES, activation='softmax')(combined)
    # model = Model(inputs=[per_model.input, api_model.input], outputs=combined_output)

    per_input = Input(shape=(PERMISSION_SIZE, PERMISSION_SIZE, 1))
    per_layer = per_input
    for idx, kernels in enumerate(nb_neurons):
        per_layer = Conv2D(kernels, kernel_size=(5, 5), strides=(1, 1), padding='same')(per_layer)
        per_layer = Activation('relu')(per_layer)

        per_layer = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(per_layer)
        per_layer = Activation('relu')(per_layer)

        per_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(per_layer)
    
    per_layer = Flatten()(per_layer)

    api_input = Input(shape=(API_SIZE, API_SIZE, 1))
    api_layer = api_input
    for idx, kernels in enumerate(nb_neurons):
        api_layer = Conv2D(kernels, kernel_size=(5, 5), strides=(1, 1), padding='same')(api_layer)
        api_layer = Activation('relu')(api_layer)

        api_layer = Conv2D(kernels, kernel_size=(3, 3), strides=(1, 1), padding='same')(api_layer)
        api_layer = Activation('relu')(api_layer)

        api_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(api_layer)
    
    api_layer = Flatten()(api_input)

    output = concatenate([per_layer, api_layer], axis=1)
    output = Dense(DENSE_UNITS, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(NB_CLASSES, activation='softmax')(output)
    model = Model(inputs=[per_input, api_input], outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, x_train, y_train, x_test, y_test, x_val, y_val):
    '''
    Train the model, return test loss
    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    '''
    model = compile_model(network)

    model.fit(x_train, y_train, 
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_val, y_val),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)
    export_path_keras = 'train_9_2020-10-25.h5'
    model.save(export_path_keras)
    return score[1]     # 1 is accuracy, 0 is loss