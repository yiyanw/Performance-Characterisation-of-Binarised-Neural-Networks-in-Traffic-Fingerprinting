import larq as lq
import numpy as np

from keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dense, Flatten, Activation, Dropout
import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model

import matplotlib.pyplot as plt

def get_first_layer(input_shape, is_binary):
    # if is_binary:
    #     return lq.layers.QuantConv2D(
    #                                 filters=32,
    #                                 kernel_size=(3, 3),
    #                                 # input_quantizer="ste_sign",
    #                                 activation="ste_sign",
    #                                 kernel_quantizer="ste_sign",
    #                                 kernel_constraint="weight_clip",
    #                                 strides=(1, 1),
    #                                 padding='same',
    #                                 data_format='channels_last',
    #                                 use_bias=False,
    #                                 )
    # else:
    #     return lq.layers.QuantConv2D(
    #                                 filters=16,
    #                                 kernel_size=(3, 3),
    #                                 strides=(1, 1),
    #                                 padding='same',
    #                                 data_format='channels_last',
    #                                 use_bias=False,
    #                                 input_shape=input_shape,
    #                                 )
    if is_binary:
        return lq.layers.QuantConv1D(
                                    filters=32,
                                    kernel_size=9,
                                    # input_quantizer="ste_sign",
                                    activation="ste_sign",
                                    kernel_quantizer="ste_sign",
                                    kernel_constraint="weight_clip",
                                    strides=1,
                                    padding='valid',
                                    data_format='channels_last',
                                    use_bias=False,
                                    )
    else:
        return lq.layers.QuantConv1D(
                                    filters=32,
                                    kernel_size=9,
                                    strides=1,
                                    padding='valid',
                                    data_format='channels_last',
                                    use_bias=False,
                                    input_shape=input_shape,
                                    )


def get_dense_layer(nb_classes, is_quantized):
    if is_quantized:
        return lq.layers.QuantDense(
                                    nb_classes,
                                    use_bias=False,
                                    activation='softmax',
                                    kernel_quantizer=lq.quantizers.DoReFa(k_bit=8, mode="weights")
                                    )
    else:
        return lq.layers.QuantDense(
                                    nb_classes,
                                    use_bias=False,
                                    activation='softmax')


def get_model(input_shape, nb_classes, parameters,
                fisrt_layer_binary=False,
                dense_layer_quantized=False,
                regularization=False):
    first_layer = get_first_layer(input_shape, fisrt_layer_binary)
    dense_layer = get_dense_layer(nb_classes, dense_layer_quantized)

    # model = tf.keras.models.Sequential([
    #     first_layer,
    #     BatchNormalization(),
    #     Activation('relu'),

    #     BatchNormalization(),
    #     lq.quantizers.SteSign(),
    #     Dropout(0.3),
    #     lq.layers.QuantConv2D(
    #         filters=32,
    #         kernel_size=(3, 3),
    #         activation="ste_sign",
    #         kernel_quantizer="ste_sign",
    #         kernel_constraint="weight_clip",
    #         strides=(1, 1),
    #         padding='same',
    #         data_format='channels_last',
    #         use_bias=False),
    #     Activation('relu'),

    #     Dropout(0.3),
    #     lq.layers.QuantConv2D(
    #         filters=32,
    #         kernel_size=(3, 3),
    #         activation="ste_sign",
    #         kernel_quantizer="ste_sign",
    #         kernel_constraint="weight_clip",
    #         strides=(1, 1),
    #         padding='same',
    #         data_format='channels_last',
    #         use_bias=False),
    #     Activation('relu'),

    #     Flatten(),
    #     dense_layer
    # ])

    model = tf.keras.models.Sequential([
        first_layer,
        BatchNormalization(),
        Activation('relu'),

        BatchNormalization(),
        lq.quantizers.SteSign(),
        # Dropout(0.3),
        lq.layers.QuantConv1D(
            filters=32,
            kernel_size=9,
            activation="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            strides=1,
            padding='valid',
            data_format='channels_last',
            use_bias=False),
        Activation('relu'),

        # Dropout(0.3),
        lq.layers.QuantConv1D(
            filters=32,
            kernel_size=9,
            activation="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            strides=1,
            padding='valid',
            data_format='channels_last',
            use_bias=False),
        Activation('relu'),

        Flatten(),
        dense_layer
    ])

    return model


def bit_to_int(data, bit_num = 8):
    # base for convert 8bit to integer
    base = []
    for i in range(bit_num-1, -1, -1):
        base.append(pow(2, i))
    # base = np.array([pow(2, 7), pow(2, 6), pow(2, 5), pow(2, 4), pow(2, 3), pow(2, 2), pow(2, 1), pow(2, 0)])
    # convert (-1 and 1) to (0 and 1)
    base = np.array(base)
    temp = np.reshape(Activation('relu').call(data), [np.shape(data)[0], -1, bit_num])
    result = np.zeros([temp.shape[0], temp.shape[1]])
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            result[i][j] = base.dot(temp[i][j])

    print(result)
    return np.reshape(result, [np.shape(data)[0], -1])



################################################# training and evaluation #######################################
def training_process(
                    X_train, y_train_cat, X_valid, y_valid_cat, 
                    input_shape, nb_classes, parameters, optimiser,
                    fisrt_layer_binary, dense_layer_quantized,
                    name,
                    show_summary=False, save_model=False):
    model = get_model(
        input_shape=input_shape, 
        nb_classes=nb_classes + 1, 
        parameters=parameters, 
        fisrt_layer_binary=fisrt_layer_binary, 
        dense_layer_quantized=dense_layer_quantized)

    print("create model")
    if save_model:
        checkpoint = [tf.keras.callbacks.ModelCheckpoint(
            'savedModels/' + name + '.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')]
    else:
        checkpoint = []

    print("compile model")
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])

    print("train model")
    history = model.fit(
            x=X_train,
            y=y_train_cat,
            batch_size=parameters.batch_size,
            epochs=parameters.nb_epochs,
            validation_data=(X_valid, y_valid_cat),
            verbose=0,  # set to no progress information, please set it to 1 if requiring progress info.
            callbacks = checkpoint)

    print("finish train model")
    if show_summary:
        lq.models.summary(model)


    return model, history


################################# evaluating #################################
def evaluate_process(model, best_model_name, X_test, y_test_cat):
    # evaluate current model
    validationAccuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print('\nfinal Keras test accuracy is : %f \n' % (100.0 * validationAccuracy[1]))

    # evaluate best model
    model.load_weights('savedModels/' + best_model_name + '.hdf5')
    bestAccuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print('\nbest Keras test accuracy is : %f \n' % (100.0 * bestAccuracy[1]))
    return validationAccuracy, bestAccuracy


def calculate_uncertainty(
                        X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat,
                        input_shape, nb_classes, parameters, optimiser, 
                        fisrt_layer_binary, dense_layer_quantized,
                        name,
                        round_num=1):

    all_best_acc = []
    for i in range(round_num):
        cur_model, history = training_process(
                                    X_train, y_train_cat, X_valid, y_valid_cat, 
                                    input_shape, nb_classes, parameters, optimiser,
                                    fisrt_layer_binary, dense_layer_quantized,
                                    name,
                                    show_summary=True, save_model=True)
        all_best_acc.append(evaluate_process(cur_model,name, X_test, y_test_cat)[1][1])


    print('all best accuracy:')
    print(all_best_acc)
    print('average best accuracy:')
    print(np.sum(all_best_acc)/round_num * 100)
    print("variance:")
    print(np.var(all_best_acc) * 100)
    print("standard deviation:")
    print(np.str(all_best_acc) * 100)




def batch_run(pre_process, name_prefix, basic_combs=True, test_first_layer_encoding=False, customize_configs=[]):
    test_combinzations = []
    if len(customize_configs) != 0:
        test_combinzations = customize_configs
    else:
        if basic_combs:
            test_combinzations += [
                ['False', False, False], 
                ['False', False, True],
                ['False', True, False],
                ['False', True, True]
            ]
        if test_first_layer_encoding:
            test_combinzations += [
                ['customized', True, False], 
                ['customized', True, True], 
                ['standard', True, False],
                ['standard', True, True],
            ]


    for each_comb in test_combinzations:
        use_thermo_encoding = each_comb[0]
        fisrt_layer_binary = each_comb[1]
        dense_layer_quantized = each_comb[2]
        X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat, input_shape, nb_classes, parameters, optimiser =\
                                                                                        pre_process(use_thermo_encoding)
        calculate_uncertainty(
                        X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat,
                        input_shape, nb_classes, parameters, optimiser, 
                        fisrt_layer_binary=fisrt_layer_binary, dense_layer_quantized=dense_layer_quantized,
                        name=name_prefix + str(use_thermo_encoding) + "_" + str(fisrt_layer_binary) + "_" + str(dense_layer_quantized),
                        round_num=5)






################################################# show information ##############################################
def counting_values(val):
    result = dict()
    for w in val:
        if w not in result:
            result[w] = 0
        result[w] += 1

    return result


def plot_key_value(d_o, name, print_xy=False, xlim_val=(-5, 5)):
    x = sorted(d_o.keys())

    sum_val = 0.0
    for i in x:
        sum_val += d_o[i]


    y = [d_o[i]/sum_val for i in x]
    plt.plot(x, y, 'o-')
    # plt.xlim(xlim_val)
    plt.savefig(name, bbox_inches="tight")
    plt.cla()
    if print_xy:
        print("x")
        print(x)
        print("y")
        print(y)


'''
# output, weights, and their corresponding counting, plot

# example:
# show_info(model, "images/AWF/", 'quant_conv2d', print_result=False)
# show_info(model, "images/AWF/", 'quant_conv2d_1', binarize_weights=True)
# show_info(model, "images/AWF/", 'quant_conv2d_2', binarize_weights=True)
# show_info(model, "images/AWF/", 'flatten', result_type=('output'), print_result=False)
# show_info(model, "images/AWF/", 'quant_dense')
'''
def show_info(model, model_preffix, layer_name, result_type=('output', 'weight'), print_result=False,
              binarize_weights=False):
    print(layer_name)
    layer = model.get_layer(layer_name)

    # outputs
    if 'output' in result_type:
        print("outputs")
        target_layer = Model(inputs=model.input, outputs=layer.output)
        out = target_layer.predict(X_train[0:2])[0].reshape(-1)
        counted_out = counting_values(out)
        plot_key_value(counted_out, model_preffix + layer_name + "_output.jpg", print_result)
        if print_result:
            print("out:")
            print(out)
            print("counted_out:")
            print(counted_out)

    # weights
    if 'weight' in result_type:
        print("weights")

        if binarize_weights:
            ws = Activation("ste_sign")(layer.weights[0].numpy())
            flatten_ws = ws.numpy().reshape(-1)
        else:
            ws = layer.weights[0]
            flatten_ws = ws.numpy().reshape(-1)
        counted_ws = counting_values(flatten_ws)
        plot_key_value(counted_ws, model_preffix + layer_name + "_weight.jpg", print_result)

        if print_result:
            print("ws:")
            print(ws)
            print("counted_ws")
            print(counted_ws)
