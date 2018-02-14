from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Dropout, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    # keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29, activation='relu'):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn = GRU(units, activation=activation, dropout=0.5,
              return_sequences=True, name='rnn0')(input_data)
    rnn = BatchNormalization(name='bn_0')(rnn)

    for l in range(recur_layers-1):
        #print(l)
        rname = 'rnn_' + str(l+1)
        bname = 'bn_' + str(l+1)
        rnn = GRU(units, activation=activation, dropout=0.5,
                  return_sequences=True, name=rname)(rnn)
        rnn = BatchNormalization(name=bname)(rnn)
        rnn = Dropout(0.5)(rnn)
        #print(rname, bname)
    
##    simp1_rnn = SimpleRNN(units, activation='relu',
##        return_sequences=True, implementation=2, name='rnn1')(input_data)
##    bn1_rnn = BatchNormalization(name='bn_rnn1')(simp1_rnn)
##    simp2_rnn = SimpleRNN(units, activation='relu',
##        return_sequences=True, implementation=2, name='rnn2')(bn1_rnn)
##    bn2_rnn = BatchNormalization(name='bn_rnn2')(simp2_rnn)
##    simp3_rnn = SimpleRNN(units, activation='relu',
##        return_sequences=True, implementation=2, name='rnn3')(bn2_rnn)
##    bn3_rnn = BatchNormalization(name='bn_rnn3')(simp3_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn1'), merge_mode='concat', weights=None)(input_data)
    rnn = BatchNormalization(name='bn_rnn')(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, layers=2, output_dim=29, dropout_rate=0.4, activation='relu'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network

    #bidir_rnn = Bidirectional(SimpleRNN(units, activation='relu',
    #    return_sequences=True, implementation=2, name='rnn1'), merge_mode='concat', weights=None)(input_data)
    rnn = Bidirectional(GRU(units, activation=activation, dropout=dropout_rate, return_sequences=True, name='rnn1'))(input_data)
    rnn = TimeDistributed(Dense(output_dim, activation=activation))(rnn)
    rnn = Dropout(dropout_rate)(rnn)

    # only allow up to 6 layers
    if layers > 6: layers = 6
    # add layers 
    if layers > 1:
        for l in range(layers-1):
            #print(l)
            lname = 'rnn' + str(l)
            rnn = Bidirectional(GRU(units, activation=activation, dropout=dropout_rate, return_sequences=True, name=lname))(rnn)
            rnn = TimeDistributed(Dense(output_dim, activation=activation))(rnn)
            rnn = Dropout(dropout_rate)(rnn)
    rnn = TimeDistributed(Dense(512, activation=activation))(rnn)
    rnn = Dropout(dropout_rate)(rnn)
    rnn = TimeDistributed(Dense(output_dim))(rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length(
    #    x, kernel_size, conv_border_mode, conv_stride)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_cnn_model(input_dim, filters, kernel_size, conv_stride,
                    conv_border_mode, units, layers=2, dropout_rate=0.4,
                    activation='relu', output_dim=29):
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network

    # Add convolutional layer
    conv = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    conv = BatchNormalization(name='bn_conv_1d')(conv)

## First try --> generates blank or nearly blank transcriptions
    #bidir_rnn = Bidirectional(SimpleRNN(units, activation='relu',
    #    return_sequences=True, implementation=2, name='rnn1'), merge_mode='concat', weights=None)(input_data)
##    rnn = Bidirectional(GRU(units, activation=activation, dropout=dropout_rate,
##                            return_sequences=True, name='rnn1'))(conv)
##    rnn = TimeDistributed(Dense(output_dim, activation=activation))(rnn)
##    rnn = Dropout(dropout_rate)(rnn)

##    # only allow up to 6 layers
##    if layers > 6: layers = 6
##    # add layers
##    if layers > 1:
##        for l in range(layers-1):
##            print(l)
##            lname = 'rnn' + str(l)
##            rnn = Bidirectional(GRU(units, activation=activation, dropout=dropout_rate,
##                                    return_sequences=True, name=lname))(rnn)
##            rnn = TimeDistributed(Dense(output_dim, activation=activation))(rnn)
##            rnn = Dropout(dropout_rate)(rnn)
##    rnn = TimeDistributed(Dense(512, activation=activation))(rnn)
##    rnn = Dropout(dropout_rate)(rnn)

## Second try

    rnn = GRU(units, activation=activation, return_sequences=True,
              name='rnn_1', dropout=dropout_rate)(conv)
    rnn = BatchNormalization(name='bt_rnn_1')(rnn)
    if layers > 1:
        for l in range(layers-1):
            #print(l)
            lname = 'rnn' + str(l)
            bname = 'bnn' + str(l)
            rnn = GRU(units, activation=activation, return_sequences=True,
                      name=lname, dropout=dropout_rate)(rnn)
            rnn = BatchNormalization(name=bname)(rnn)
        ##
    rnn = TimeDistributed(Dense(512, activation=activation))(rnn)
    rnn = Dropout(dropout_rate)(rnn)
    rnn = TimeDistributed(Dense(output_dim))(rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)

    print(model.summary())
    return model
