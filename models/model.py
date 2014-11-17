import numpy

from fastor.layers import layers, cc_layers
import fastor.models

class ReinforcementModel(fastor.models.ReinforcementModel):
    batch = 32
    num_frames = 4
    img_dims = (84, 84)
    dims_x, dims_y = img_dims
    action_dims = 4

    input = cc_layers.CudaConvnetInput2DLayer(batch, num_frames, dims_x, dims_y)    
    
    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(8*8*3) # was = 0.25  
    winit2 = k/numpy.sqrt(4*4*16)
    binit = 1.0

    nonlinearity = layers.rectify

    conv1 = cc_layers.CudaConvnetConv2DLayer(
        input, 
        n_filters=16,
        filter_size=8,
        stride=4, 
        weights_std=winit1,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=0)
    conv2 = cc_layers.CudaConvnetConv2DLayer(
        conv1,
        n_filters=32,
        filter_size=4,
        stride=2,
        weights_std=winit2,
        init_bias_value=binit,
        nonlinearity=nonlinearity,
        pad=0)


    winitD1 = k/numpy.sqrt(numpy.prod(conv2.get_output_shape()))
    winitD2 = k/numpy.sqrt(256)

    conv2_shuffle = cc_layers.ShuffleC01BToBC01Layer(conv2)    
    fc3 = layers.DenseLayer(
        conv2_shuffle,
        n_outputs = 256,
        weights_std=winitD1,
        init_bias_value=1.0,
        nonlinearity=layers.rectify,
        dropout=0.5)
    output = layers.DenseLayer(
        fc3,
        n_outputs=action_dims,
        weights_std=winitD2,
        init_bias_value=binit,
        nonlinearity=layers.identity)
