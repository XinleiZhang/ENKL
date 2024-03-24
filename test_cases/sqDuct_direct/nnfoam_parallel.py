# Copyright 2018 Virginia Polytechnic Institute and State University.
""" Dynamic model for OpenFOAM eddy viscosity nutFoam solver. """

# standard library imports
import os
import shutil
import subprocess
import multiprocessing

# third party imports
import numpy as np
import scipy.sparse as sp
import yaml

# local imports
from dafi import PhysicsModel
from dafi import random_field as rf
from dafi.random_field import foam


import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import neuralnet
import gradient_descent as gd
import regularization as reg
import data_preproc as preproc
import cost
from get_inputs import get_inputs


import pdb

TENSORDIM = 9
TENSORSQRTDIM = 3
DEVSYMTENSORDIM = 5
DEVSYMTENSOR_INDEX = [0,1,2,4,5]
NBASISTENSORS = 10
NSCALARINVARIANTS = 5

VECTORDIM = 3
nscalar_invariants = 2
nbasis_tensors = 4
nhlayers = 2
nnodes = 5
alpha = 0

n1 = 18
n2 = 48
n3 = 46
# for 0.2
# n1 = 26
# n2 = 46
## CREATE NN


class Model(PhysicsModel):
    """ Dynamic model for OpenFoam Reynolds stress nutFoam solver.

    The eddy viscosity field (nu_t) is infered by observing the
    velocity field (U). Nut is modeled as a random field with lognormal
    distribution and median value equal to the baseline (prior) nut
    field.
    """

    def __init__(self, inputs_dafi, inputs_model):
        # get required DAFI inputs.
        self.nsamples = inputs_dafi['nsamples']
        max_iterations = inputs_dafi['max_iterations']
        self.analysis_to_obs = inputs_dafi['analysis_to_obs']

        # read input file
        self.foam_case = inputs_model['foam_case']
        iteration_nstep = inputs_model['iteration_nstep']
        self.foam_timedir = str(iteration_nstep)

        nweights = inputs_model.get('nweights', None)
        self.ncpu = inputs_model.get('ncpu', 20)
        self.rel_stddev = inputs_model.get('rel_stddev', 0.5)
        self.abs_stddev = inputs_model.get('abs_stddev', 0.5)
        self.obs_rel_std = inputs_model.get('obs_rel_std', 0.001)
        self.obs_abs_std = inputs_model.get('obs_abs_std', 0.0001)

        obs_file = inputs_model['obs_file']
        # obs_err_file = inputs_model['obs_err_file']
        # obs_mat_file = inputs_model['obs_mat_file']

        weight_baseline_file = inputs_model['weight_baseline_file']

        # required attributes
        self.name = 'NN parameterized RANS model'

        # results directory
        self.results_dir = 'results_ensemble'

        # counter
        self.da_iteration = -1

        
        # NN architecture
        self.nscalar_invariants = inputs_model.get('nscalar_invariants', NSCALARINVARIANTS)
        self.nbasis_tensors = inputs_model.get('nbasis_tensors', NBASISTENSORS)
        nhlayers = inputs_model.get('nhlayers', 10)
        nnodes = inputs_model.get('nnodes', 10)
        alpha = inputs_model.get('alpha', 0.0)
        self.nn = neuralnet.NN(self.nscalar_invariants, self.nbasis_tensors,nhlayers, nnodes, alpha)
        
        # initial g
        self.g_init  = np.array(inputs_model.get('g_init', [0.0]*self.nbasis_tensors))
        self.g_scale = inputs_model.get('g_scale', 1.0)

        # data pre-processing
        self.preproc_class = inputs_model.get('preproc_class', None)

        # debug
        self.fixed_inputs  = inputs_model.get('fixed_inputs', True)

        parallel = inputs_model.get('parallel', True)



        # call Tensorflow to get initilazation messages out of the way
        with tf.GradientTape(persistent=True) as tape:
            gtmp = self.nn(np.zeros([1, self.nscalar_invariants]))
        _ = tape.jacobian(gtmp, self.nn.trainable_variables, experimental_use_pfor=False)

        # initial weights
        self.w_init = np.loadtxt(weight_baseline_file)#.mean(axis=1) # np.array([])
        # self.w_init = np.loadtxt('./results_dafi/t_0/xf/xf_27') # np.array([])

        self.nbasis = self.nbasis_tensors
        self.nstate = len(self.w_init)

        self.w_shapes = neuralnet.weights_shape(self.nn.trainable_variables)

        # print NN summary
        print('\n' + '#'*80 + '\nCreated NN:' +
            f'\n  Number of scalar invariants: {self.nscalar_invariants}' +
            f'\n  Number of basis tensors: {self.nbasis_tensors}' +
            f'\n  Number of trainable parameters: {self.nn.count_params()}' +
            '\n' + '#'*80)

        # get the preprocesing class
        if self.preproc_class is not None:
            self.PreProc = getattr(preproc, self.preproc_class)

        self.first_iter = -1

        # calculate inputs
        # initialize preprocessing instance
        if os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)
        self.preprocess_data = self.PreProc()

        data_dir = 'nnfoam_inputs/data'
        self.theta = np.load(os.path.join(data_dir, 'scalar_invariants.npy'))[:, :nscalar_invariants]

        self.obs = shih_quadratic(self.theta)[:,0] #.flatten() #np.concatenate([obs_Uy, obs_Uz])
        self.obs_error = np.diag((self.obs_rel_std * self.obs + self.obs_abs_std)**2)
        self.nstate_obs = len(self.obs)

        # create sample directories
        # sample_dirs = []
        # for isample in range(self.nsamples+1):
        #     sample_dir = self._sample_dir(isample)
        #     sample_dirs.append(sample_dir)
        #     # TODO foam.copyfoam(, post='') - copies system, constant, 0
        #     shutil.copytree(self.foam_case, sample_dir)
        #     foam.write_controlDict(
        #         self.control_list, self.foam_info['foam_version'],
        #         self.foam_info['website'], ofcase=sample_dir)
        # self.sample_dirs = sample_dirs
        # pdb.set_trace()


    def __str__(self):
        return 'Dynamic model for nutFoam eddy viscosity solver.'

    # required methods
    def generate_ensemble(self):
        """ Return states at the first data assimilation time-step.

        Creates the OpenFOAM case directories for each sample, creates
        samples of eddy viscosity (nut) based on samples of the KL modes
        coefficients (state) and writes nut field files. Returns the
        coefficients of KL modes for each sample.
        """

        # update X (nut)
        w = np.zeros([self.nstate, self.nsamples+1])
        for i in range(self.nstate):
            w[i, :-1] = self.w_init[i] + np.random.normal(0,
                abs(self.w_init[i] * self.rel_stddev + self.abs_stddev)
                , self.nsamples)
        
        w[:,-1] = w[:,:-1].mean(axis=1)
        return w

    def state_to_observation(self, state_vec):
        """ Map the states to observation space (from X to HX).

        Modifies the OpenFOAM cases to use nu_t reconstructed from the
        specified coeffiecients. Runs OpenFOAM, and returns the
        velocities at the observation locations.
        """
        self.da_iteration += 1

        # set weights
        w = state_vec.copy()
        time_dir = f'{self.da_iteration:d}'
        gsamps = []

        ts = time.time()
        for isamp in range(w.shape[1]):

            # for i in range(self.nsamples):
            weight = state_vec[:, isamp]
            w_reshape = neuralnet.reshape_weights(weight, self.w_shapes)
            self.nn.set_weights(w_reshape)
            # evaluate NN: cost and gradient
            with tf.GradientTape(persistent=True) as tape:
                g = self.nn(self.theta) * self.g_scale + self.g_init

            g = np.array(g)[:,0]#.flatten()

            gsamps.append(g)

        state_in_obs = np.array(gsamps).T

        return state_in_obs

    def get_obs(self, time):
        """ Return the observation and error matrix.
        """
        return self.obs, self.obs_error

# shih_quadratic model: theta > g function
def shih_quadratic(theta):
    def g1(theta):
        num = -2./3.
        denom = 1.25 + np.sqrt(2 * theta[:, 0]) + 0.9 * np.sqrt(-2 * theta[:, 1])
        return num/denom

    def g234(theta, coeff):
        return coeff / (1000. + (2 * theta[:, 0])**(3./2.))

    g = np.empty([len(theta), nbasis_tensors])
    g[:, 0] = g1(theta)
    for i, c in enumerate([7.5, 1.5, -9.5]):
        g[:, i+1] = g234(theta, c)
    return g

