import os

#set threads of tf = 1
os.environ["TF_NUM_INTEROP_THREADS"] = "1"  #
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"  # 
# set threads of openmp = 1
os.environ["OMP_NUM_THREADS"] = "1"  #

import numpy as np
import tensorflow as tf
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

f = open('python_module.log', 'w+')
print('_________________________________________________________________', file=f)
print('Computing function of scalar invariants from Python module', file=f)
print('Tensorflow version', tf.__version__, file=f)
print('_________________________________________________________________', file=f)

# tf.keras.backend.clear_session()
# tf.config.threading.set_intra_op_parallelism_threads(2)
# load model

model_path = '../../NN-PRE-TRAIN/my_test_model.h5'
model = tf.keras.models.load_model(model_path)

# load weights
# get weights flatten
weights_flatten = np.loadtxt('nn_weights_flatten.dat')

# get model shape
shapes = []
for iw in model.trainable_variables:
    shapes.append(iw.shape)

# shapes to sizes
sizes = []
for shape in shapes:
    isize = 1
    for ishape in shape:
        isize *= ishape
    sizes.append(isize)

# reshape weights
w_reshaped = []
i = 0
for shape, size in zip(shapes, sizes):
    w_reshaped.append(weights_flatten[i:i + size].reshape(shape))
    i += size
    # print(w_reshaped)
model.set_weights(w_reshaped)

print(model.get_weights(), file=f)
print('Neural-network weights loaded successfully', file=f)



def ml_func(array):
    t1 = time.time()
    # print(np.shape(array))
    array_scaled = np.zeros_like(array)

    theta1_min = array[:, 0].min()
    theta1_max = array[:, 0].max()
    theta2_min = array[:, 1].min()
    theta2_max = array[:, 1].max()

    array_scaled[:, 0] = (array[:, 0] - theta1_min) / (theta1_max - theta1_min)
    array_scaled[:, 1] = (array[:, 1] - theta2_min) / (theta2_max - theta2_min)

    g_ = model(array_scaled, training=False)

    g = np.array(g_).reshape(-1, 4).astype('double')
    # print(g)
    scale = [0.5,0.05,0.05,0.05]#[0.1,0.001,0.001,0.001]
    init = [-0.09,0,0,0]#[-0.09,1e-4,1e-4,1e-4]
    for i in range(g.shape[1]):
        g[:,i] = scale[i]*g[:,i] + init[i]

    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if j == 0 and g[i, j] > -0.0: g[i, j] = -0.0

    return g


if __name__ == "__main__":
    print('in main')
