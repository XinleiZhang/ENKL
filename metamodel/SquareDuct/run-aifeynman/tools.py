# case specification
dafiCase = '../case-1/'
sampleId = 50
timeDir = 100
################################################################
# load neural network
import time
import tensorflow as tf
import numpy as np
import os

# set threads of tf = 1
os.environ["TF_NUM_INTEROP_THREADS"] = "1"  #
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"  #
# set threads of openmp = 1
os.environ["OMP_NUM_THREADS"] = "1"  #

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

f = open('python_module.log', 'w+')
print('_________________________________________________________________',
      file=f)
print('Computing function of scalar invariants from Python module', file=f)
print('Tensorflow version', tf.__version__, file=f)
print('_________________________________________________________________',
      file=f)

# tf.keras.backend.clear_session()
# tf.config.threading.set_intra_op_parallelism_threads(2)
# load model

model_path = '../NN-pre-train/nn_model_235.h5'
model = tf.keras.models.load_model(model_path)
# model.summary()

# load weights
# get weights flatten
weights_flatten = np.loadtxt(
    os.path.join(dafiCase, 'results_ensemble', f'sample_{sampleId}',
                 'nn_weights_flatten.dat'))
# weights_flatten = np.loadtxt('/home/wuct/dafi/pehills-NN-closure/inputs/w.0')

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
    # print(np.shape(array))
    array_scaled = np.zeros_like(array)
    t12 = np.loadtxt(f'theta-range-{sampleId}.csv', delimiter=',')

    array_scaled[:, 0] = (array[:, 0] - t12[0, 0]) / (t12[0, 1] - t12[0, 0])
    array_scaled[:, 1] = (array[:, 1] - t12[1, 0]) / (t12[1, 1] - t12[1, 0])
    array_scaled[:, 2] = (array[:, 2] - t12[1, 0]) / (t12[1, 1] - t12[1, 0])
    array_scaled[:, 3] = (array[:, 3] - t12[1, 0]) / (t12[1, 1] - t12[1, 0])
    array_scaled[:, 4] = (array[:, 4] - t12[1, 0]) / (t12[1, 1] - t12[1, 0])
    array_scaled[:, 5] = (array[:, 5] - t12[1, 0]) / (t12[1, 1] - t12[1, 0])

    g_ = model(array_scaled, training=False)

    g = np.array(g_).reshape(-1, 5).astype('double')
    # print(g)
    scale = [1.e-1, 1.e-3, 1.e-3, 1.e-3, 0]  # [0.1,0.001,0.001,0.001]
    init = [-0.09, 0, 0, 0, 0]  # [-0.09,1e-4,1e-4,1e-4]
    for i in range(g.shape[1]):
        # scale[i]*g[:, i] + (1.0 - scale[i])*init[i]
        g[:, i] = scale[i] * g[:, i] + init[i]

    # print(g)

    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if j == 0 and g[i, j] > -0.0:
                g[i, j] = -0.0
            # if j == 1 and g[i, j] > 0.01: g[i, j] = 0.01
            # if j == 2 and g[i, j] < -0.01: g[i, j] = -0.01
            # if j == 3 and g[i, j] < -0.01: g[i, j] = -0.01
    # print('using tensorflow time', time.time() - t1)

    return g


def g1_nn(theta_):
    g_ = ml_func(theta_)
    # perturbed_data = g_[:, 0] + np.random.normal(0, 1e-8, np.shape(g_[:, 0]))
    return g_[:, 0]  #+ perturbed_data


def g1_SHIH(theta_):
    if (theta_[:, 0] < 0).any():
        print('x1 need be larger than zero')
        return
    if (theta_[:, 1] > 0).any():
        print('x2 need be less than zero')
        return
    return -(2.0 / 3.0) / (1.25 + np.sqrt(2 * theta_[:, 0]) +
                           0.9 * np.sqrt(-2 * theta_[:, 1]))


def g1_Shih(x1, x2):
    if (x1 < 0).any():
        print('x1 need be larger than zero')
        return
    if (x2 > 0).any():
        print('x2 need be less than zero')
        return
    return -(2.0 / 3.0) / (1.25 + np.sqrt(2 * x1) + 0.9 * np.sqrt(-2 * x2))


def g2_Shih(x1):
    if (x1 < 0).any():
        print('x1 need be larger than zero')
        return
    return 7.5 / (1000 + np.power(2 * x1, 1.5))


def g3_Shih(x1):
    if (x1 < 0).any():
        print('x1 need be larger than zero')
        return
    return 1.5 / (1000 + np.power(2 * x1, 1.5))


def g4_Shih(x1):
    if (x1 < 0).any():
        print('x1 need be larger than zero')
        return
    return -9.5 / (1000 + np.power(2 * x1, 1.5))
