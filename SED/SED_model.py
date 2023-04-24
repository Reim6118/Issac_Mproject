import tensorflow.keras
import tensorflow 
import keras.api._v2.keras as keras
import keras
def weighted_binary_crossentropy(zero_weight, one_weight):
    """
    Loss with support for specifying class weights
    """
    import tensorflow.keras.backend as K
    
    def weighted_binary_crossentropy(y_true, y_pred):

        # standard cross entropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # apply weighting
        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def build_sednet(input_shape, filters=128, cnn_pooling=(5, 2, 2), rnn_units=(32, 32), dense_units=(32,), n_classes=1, dropout=0.5):
    """
    SEDnet type model

    Based https://github.com/sharathadavanne/sed-crnn/blob/master/sed.py
    """
    
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Bidirectional, Conv2D, BatchNormalization, Activation, \
            Dense, MaxPooling2D, Dropout, Permute, Reshape, GRU, TimeDistributed
    
    spec_start = Input(shape=(input_shape[-3], input_shape[-2], input_shape[-1]))
    spec_x = spec_start
    for i, pool in enumerate(cnn_pooling):
        spec_x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, pool))(spec_x)
        spec_x = Dropout(dropout)(spec_x)
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((input_shape[-3], -1))(spec_x)

    for units in rnn_units:
        spec_x = Bidirectional(
            GRU(units, activation='tanh', dropout=dropout, recurrent_dropout=dropout, return_sequences=True),
            merge_mode='mul')(spec_x)

    for units in dense_units:
        spec_x = TimeDistributed(Dense(units))(spec_x)
        spec_x = Dropout(dropout)(spec_x)

    spec_x = TimeDistributed(Dense(n_classes))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)
    
    model = Model(inputs=spec_start, outputs=out)
    
    return model
   
def build_model(input_shape, dropout=0.5, lr=0.01, class_weights=None):
    
    # Model
    model = build_sednet(input_shape, n_classes=1,
                         filters=10,
                         cnn_pooling=[2, 2, 2],
                         rnn_units=[5, 5],
                         dense_units=[16],
                         dropout=dropout,
                        )

    # Metrics
    pr_auc = tensorflow.keras.metrics.AUC(num_thresholds=200, curve="PR", name='pr_auc')
    precision = tensorflow.keras.metrics.Precision(name='precision')
    recall = tensorflow.keras.metrics.Recall(name='recall')
    
    # Loss
    if class_weights is None:
        loss = tensorflow.keras.losses.BinaryCrossentropy()
    else:
        loss = weighted_binary_crossentropy(*class_weights)
       
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lr),
                  loss=loss,
                  metrics=[pr_auc, precision, recall],
    )
    return model

