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
        

        self.timeprecision = 6
        
        # initial g
        self.g_init  = np.array(inputs_model.get('g_init', [0.0]*self.nbasis_tensors))
        self.g_scale = inputs_model.get('g_scale', 1.0)

        # data pre-processing
        self.preproc_class = inputs_model.get('preproc_class', None)

        # debug
        self.fixed_inputs  = inputs_model.get('fixed_inputs', True)

        parallel = inputs_model.get('parallel', True)

        ## CREATE NN
        self.nn = tf.keras.models.load_model('NN-PRE-TRAIN/my_test_model.h5')
        
        self.nn.summary()
        
        # initial weights
        self.w_init = np.loadtxt(weight_baseline_file) 

        self.nbasis = self.nbasis_tensors
        self.nstate = len(self.w_init)

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

        # observations
        # read observations
        U_obs = foam.read_vector_field(obs_file)
        Ux = U_obs[:, 0] # 0.5
        Uy = U_obs[:, 1] * 1000 # 500
        Uz = U_obs[:, 2] * 1000 # 500
        self.obs = np.concatenate([Ux, Uy, Uz])
        self.obs_error = np.diag(self.obs_rel_std * abs(self.obs) + self.obs_abs_std)
        self.nstate_obs = len(self.obs)

        # create sample directories // one more mean sample
        sample_dirs = []
        for isample in range(self.nsamples+1):
            sample_dir = self._sample_dir(isample)
            sample_dirs.append(sample_dir)
            # TODO foam.copyfoam(, post='') - copies system, constant, 0
            shutil.copytree(self.foam_case, sample_dir)

        self.sample_dirs = sample_dirs
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
        
        w = state_vec.copy()
        time_dir = f'{self.da_iteration+1:d}'
        self.preprocess_data = self.PreProc()
        ts = time.time()
        
        # print(w.shape[1])
        for isamp in range(w.shape[1]):
            modelPath = os.path.join(self._sample_dir(isamp), 'nn_weights_flatten.dat')
            np.savetxt(modelPath, w[:, isamp])
        
        parallel = multiprocessing.Pool(self.ncpu)
        inputs = [
            (self._sample_dir(i), self.da_iteration,
                self.timeprecision) for i in range(w.shape[1])]
        _ = parallel.starmap_async(_run_foam, inputs)
        parallel.close()
        parallel.join()


        state_in_obs = np.empty([self.nstate_obs, w.shape[1]])
        for isample in range(w.shape[1]):
            file = os.path.join(self._sample_dir(isample), time_dir, 'U')
            U = foam.read_vector_field(file)
            Ux = U[:, 0] # * 0.5
            Uy = U[:, 1] * 1000 #500
            Uz = U[:, 2] * 1000 #500
            state_in_obs[:, isample] = np.concatenate([Ux, Uy, Uz])
        return state_in_obs

    def get_obs(self, time):
        """ Return the observation and error matrix.
        """
        return self.obs, self.obs_error

    def clean(self, loop):
        if loop == 'iter' and self.analysis_to_obs:
            for isamp in range(self.nsamples):
                dir = os.path.join(self._sample_dir(isamp),
                                   f'{self.da_iteration + 1:d}')
                shutil.rmtree(dir)

    # internal methods
    def _sample_dir(self, isample):
        "Return name of the sample's directory. "
        return os.path.join(self.results_dir, f'sample_{isample:d}')

    def _modify_foam_case(self, g, step, foam_dir=None):
        for i, g_data in enumerate(self.g_data_list):
            g_data['internal_field']['value'] = g[:, i]
            if foam_dir is not None:
                g_data['file'] = os.path.join(foam_dir, str(step), f'g{i+1}')
            _ = rf.foam.write_field_file(**g_data)

## Gradient: analytic dTau/dg
def _get_dadg(tensors, tke):
    tke = np.expand_dims(np.squeeze(tke), axis=(1, 2))
    return 2.0*tke*tensors

def _clean_foam(foam_dir):
    bash_command = './clean > /dev/null'
    bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)

def _run_foam_init(foam_dir, iteration, timeprecision):
    bash_command = './run > /dev/null'
    bash_command = 'cd ' + foam_dir + ';' + bash_command + '; cd -'
    return subprocess.call(bash_command, shell=True, stdout=subprocess.DEVNULL)

def _run_foam(foam_dir, iteration, timeprecision):

    # run foam
    os.chdir(foam_dir)
    solver = 'PysimpleFoam'
    logfile = os.path.join(solver + '.log')
    bash_command = f'{solver} > {logfile}'
    subprocess.call(bash_command, shell=True)
    logfile = os.path.join('gradU.log')
    bash_command = f"postProcess -func 'grad(U)' " + f"> {logfile}" # 2>&1'
    subprocess.call(bash_command, shell=True)
    
    os.chdir('../../')
