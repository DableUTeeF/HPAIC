from keras import applications as ka, layers as kl, models as km
from keras import backend as K
bn_axis = 3


def densenet201():
    base_model = ka.densenet.DenseNet201(False, pooling='avg')
    inp = kl.Input((224, 224, 4))
    x = kl.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(inp)
    x = kl.BatchNormalization(epsilon=1.001e-5,
                           name='conv1/bn')(x)
    base_model.layers.pop(0)
    base_model.layers.pop(0)
    base_model.summary()
    base_model = base_model(inp)
    avg = base_model.get_layer('avg_pool').output
    output = kl.Dense(28, activation='sigmoid')(avg)
    return km.Model(inp, output)


def densenet(blocks=[6, 12, 48, 32], classes=28):
    img_input = kl.Input((224, 224, 4))
    x = kl.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = kl.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = kl.BatchNormalization(epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = kl.Activation('relu', name='conv1/relu')(x)
    x = kl.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = kl.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = kl.BatchNormalization(epsilon=1.001e-5,
                           name='bn')(x)

    x = kl.GlobalAveragePooling2D(name='avg_pool')(x)
    x = kl.Dense(classes, activation='softmax', name='fc1000')(x)
    return km.Model(img_input, x)


def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    x = kl.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = kl.Activation('relu', name=name + '_relu')(x)
    x = kl.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = kl.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    x1 = kl.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = kl.Activation('relu', name=name + '_0_relu')(x1)
    x1 = kl.Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = kl.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = kl.Activation('relu', name=name + '_1_relu')(x1)
    x1 = kl.Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = kl.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x
