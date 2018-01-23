import numpy as np

import keras.layers

from caffe2.python import core, model_helper, workspace, brew
from caffe2.proto import caffe2_pb2

def copy_model(keras_model):
    
    arg_scope = {'order': 'NCHW', 'use_cudnn': False}
    caffe2_model = model_helper.ModelHelper(name='model', arg_scope=arg_scope)
    
    prev_name = 'in'
    prev_shape = None
    
    for keras_layer in keras_model.layers:
        config = keras_layer.get_config()
        
        if 'data_format' in config:
            if config['data_format'] != 'channels_first':
                raise ValueError('The only supported data_format is channels_first')
                
        if prev_shape is None:
            shape = config['batch_input_shape']
            shape = shape[1:]
        else:
            shape = prev_shape
        
        name = config['name']

        if isinstance(keras_layer, keras.layers.Conv2D):
            dim_in = shape[0]
            dim_out = config['filters']
            kernel = config['kernel_size'][0]
            stride = config['strides'][0]
            
            brew.conv(caffe2_model,
                      prev_name,
                      name,
                      dim_in=dim_in,
                      dim_out=dim_out,
                      kernel=kernel,
                      stride=stride)
            
            if config['activation'] != 'relu':
                raise ValueError('The only supported activation for conv layer is relu')
            brew.relu(caffe2_model, name, name)
            
            prev_shape = [None, None, None]
            prev_shape[0] = dim_out
            prev_shape[1] =  prev_shape[2] = int((shape[2] - kernel) / stride) + 1
            prev_name = name
            
        elif isinstance(keras_layer, keras.layers.BatchNormalization):
            if not isinstance(shape, list):
                continue

            dim_in = shape[0]
            epsilon = config['epsilon']
            momentum = config['momentum']
            brew.spatial_bn(caffe2_model,
                            prev_name,
                            name, 
                            dim_in=dim_in, 
                            epsilon=epsilon, 
                            momentum=momentum,
                            is_test=True)
            prev_name = name
            
        elif isinstance(keras_layer, keras.layers.MaxPooling2D):
            kernel = config['pool_size'][0]
            stride = config['strides'][0]
            brew.max_pool(caffe2_model,
                          prev_name,
                          name,
                          kernel=kernel,
                          stride=stride)
            prev_shape[1] =  prev_shape[2] = int((shape[2] - kernel) / stride) + 1
            prev_name = name
            
        elif isinstance(keras_layer, keras.layers.Flatten):
            prev_shape = shape[0] * shape[1] * shape[2]
            
        elif isinstance(keras_layer, keras.layers.Dense):
            dim_in = shape
            dim_out = config['units']
            
            brew.fc(caffe2_model,
                    prev_name,
                    name,
                    dim_in=dim_in,
                    dim_out=dim_out)
            
            activation = config['activation']
            if activation == 'relu':
                brew.relu(caffe2_model, name, name)
            elif activation == 'softmax':
                brew.softmax(caffe2_model, name, name)
            elif activation == 'linear':
                pass # Don't need to apply anything
            else:
                raise ValueError('The only supported activations for fc layer are relu and softmax')
            prev_shape = dim_out
            prev_name = name
            
    return caffe2_model

def set_weights(keras_model, caffe2_model):
    
    for keras_layer in keras_model.layers:
        
        name = keras_layer.get_config()['name']  

        if isinstance(keras_layer, keras.layers.Conv2D):            
            w = keras_layer.get_weights()[0].transpose((3, 2, 0, 1))
            b = keras_layer.get_weights()[1]
            workspace.FeedBlob(name + '_w', w)
            workspace.FeedBlob(name + '_b', b)
            
        elif isinstance(keras_layer, keras.layers.BatchNormalization):
            s = keras_layer.get_weights()[0]
            b = keras_layer.get_weights()[1]
            rm = keras_layer.get_weights()[2]
            riv = keras_layer.get_weights()[3]
            
            workspace.FeedBlob(name + '_s', s)
            workspace.FeedBlob(name + '_b', b)
            workspace.FeedBlob(name + '_rm', rm)
            workspace.FeedBlob(name + '_riv', riv)
            
            # Add rm and riv parameters of spatial_bn layers to params list of the model
            caffe2_model.params.append(workspace.StringifyBlobName(name + '_rm'))
            caffe2_model.params.append(workspace.StringifyBlobName(name + '_riv'))

        elif isinstance(keras_layer, keras.layers.Dense):     
            w = keras_layer.get_weights()[0].transpose()
            b = keras_layer.get_weights()[1]
            workspace.FeedBlob(name + '_w', w)
            workspace.FeedBlob(name + '_b', b)
        

def keras_to_caffe2(keras_model):
    
    caffe2_model = copy_model(keras_model)
    
    input_shape = list(keras_model.layers[0].get_config()['batch_input_shape'])
    input_shape[0] = 1
    
    np_data = np.zeros(input_shape, dtype=np.float32)
    workspace.FeedBlob('in', np_data)

    workspace.RunNetOnce(caffe2_model.param_init_net)
    workspace.CreateNet(caffe2_model.net)
    
    set_weights(keras_model, caffe2_model)
    
    return caffe2_model