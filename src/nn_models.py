"""Reference-inspired Keras classifiers. Training always requires the local GPU."""
from tensorflow import keras

def compile_model(model, learning_rate=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'],jit_compile=False)
    return model

def build_ffnn(input_shape, n_classes=10, learning_rate=1e-3):
    inputs=keras.Input(shape=input_shape,dtype='float32')
    x=keras.layers.Flatten()(inputs)
    for units in (1024,512,256,128,64):
        x=keras.layers.Dense(units,activation='relu')(x)
    outputs=keras.layers.Dense(n_classes,activation='softmax')(x)
    return compile_model(keras.Model(inputs,outputs,name='ffnn'),learning_rate)

def build_cnn(input_shape,input_type,n_classes=10,learning_rate=1e-3,dropout=.25):
    layers=keras.layers
    if input_type=='mel':
        conv,pool,kernels,pool_size=layers.Conv2D,layers.MaxPooling2D,((4,4),(3,3),(3,3)),(2,2)
    elif input_type=='raw':
        conv,pool,kernels,pool_size=layers.Conv1D,layers.MaxPooling1D,(4,3,3),8
    else: raise ValueError('input_type must be raw or mel')
    inputs=keras.Input(shape=input_shape,dtype='float32')
    x=inputs
    for filters,kernel in zip((32,64,128),kernels):
        x=conv(filters,kernel,padding='same',kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x=layers.BatchNormalization()(x)
        x=layers.ReLU()(x)
        x=pool(pool_size=pool_size)(x)
        x=layers.Dropout(dropout)(x)
    x=layers.Flatten()(x)
    x=layers.Dense(256,activation='relu')(x)
    x=layers.Dense(64,activation='relu')(x)
    outputs=layers.Dense(n_classes,activation='softmax')(x)
    return compile_model(keras.Model(inputs,outputs,name=f'cnn_{input_type}'),learning_rate)
