import tensorflow as tf
from tensorflow import pad
from tensorflow.keras.layers import Layer, Conv2D, ReLU, Add

from tensorflow_addons.layers import InstanceNormalization


# ReflectionPadding-Implementation from:
# https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/
'''
  2D Reflection Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
'''
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')
    
    
    
    
def resnet_convBlock(n_filters):
    result = tf.keras.Sequential()
    
    layers = [
        ReflectionPadding2D(padding=(1, 1)),
        Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding="valid"),
        InstanceNormalization(axis=-1),
        ReLU(),
        
        ReflectionPadding2D(padding=(1, 1)),
        Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding="valid"),
        InstanceNormalization(axis=-1)
    ]
    
    for layer in layers:
        result.add(layer)
        
    return result

def resnet_block(input_layer, n_filters):
    t = resnet_convBlock(n_filters)(input_layer)
    output = Add()([t, input_layer])
    return output
    
    
    