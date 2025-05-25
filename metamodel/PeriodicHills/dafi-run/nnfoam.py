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
DEVSYMTENSOR_INDEX = [0, 1, 2, 4, 5]
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

        weight_baseline_file = inputs_model['weight_baseline_file']

        # required attributes
        self.name = 'NN parameterized RANS model'

        # results directory
        self.results_dir = 'results_ensemble'

        # counter
        self.da_iteration = -1

        # # control dictionary
        self.timeprecision = 6

        # data pre-processing
        self.preproc_class = inputs_model.get('preproc_class', None)

        # debug
        self.fixed_inputs = inputs_model.get('fixed_inputs', True)

        parallel = inputs_model.get('parallel', True)

        ## CREATE NN

        # initial weights
        self.w_init = np.loadtxt(weight_baseline_file)
        # self.w_init = np.loadtxt('./results_dafi/t_0/xf/xf_27') # np.array([])

        # self.nbasis = self.nbasis_tensors
        self.nstate = len(self.w_init)

        # get the preprocesing class
        if self.preproc_class is not None:
            self.PreProc = getattr(preproc, self.preproc_class)

        # calculate inputs
        # initialize preprocessing instance
        if os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)

        # observations
        # read experimental observations

        expArray = np.loadtxt(obs_file + '/Array-dns.dat')

        # pdb.set_trace()
        self.obs = expArray
        self.obs_error = np.diag(self.obs_rel_std * abs(self.obs) +
                                 self.obs_abs_std)
        # self.obs_error = np.diag((self.obs_rel_std * self.obs + self.obs_abs_std)**2)
        self.nstate_obs = len(self.obs)

        # create sample directories
        sample_dirs = []
        for isample in range(self.nsamples):
            sample_dir = self._sample_dir(isample)
            sample_dirs.append(sample_dir)
            # TODO foam.copyfoam(, post='') - copies system, constant, 0
            shutil.copytree(self.foam_case, sample_dir)
            # foam.write_controlDict(self.control_list,
            #                        self.foam_info['foam_version'],
            #                        self.foam_info['website'],
            #                        ofcase=sample_dir)
        self.sample_dirs = sample_dirs

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
        w = np.zeros([self.nstate, self.nsamples])
        for i in range(self.nstate):
            np.random.seed(0)
            w[i, :] = self.w_init[i] + np.random.normal(
                0, abs(self.w_init[i] * self.rel_stddev + self.abs_stddev),
                self.nsamples)
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

        gsamps = []
        self.preprocess_data = self.PreProc()
        ts = time.time()
        for isamp in range(self.nsamples):
            modelPath = os.path.join(self._sample_dir(isamp),
                                     'nn_weights_flatten.dat')
            # print('saving neural network to', modelPath)
            np.savetxt(modelPath, w[:, isamp])
        print(self.da_iteration, 'nn_weight had saved')

        # print('here run')

        parallel = multiprocessing.Pool(self.ncpu)
        inputs = [(self._sample_dir(i), self.da_iteration, self.timeprecision)
                  for i in range(self.nsamples)]
        _ = parallel.starmap(_run_foam, inputs)
        parallel.close()

        # get HX
        # U_ref = 14.0

        time_start = 2
        time_step = 1

        time_dir = f'{((self.da_iteration+1)*time_step+time_start):g}'
        # print('state in obs OpenFOAM time', time_dir)
        state_in_obs = np.empty([self.nstate_obs, self.nsamples])
        for isample in range(self.nsamples):

            file = os.path.join(self._sample_dir(isample), 'postProcessing',
                                'sampleDict', time_dir)

            state_in_obs[:, isample] = np.loadtxt(file + '/Array-obs.dat')

        print('state_in_obs using', time.time() - ts)
        return state_in_obs

    def get_obs(self, time):
        # print('in get_obs')
        """ Return the observation and error matrix.
        """
        return self.obs, self.obs_error

    def clean(self, loop):
        print('in clean')
        if loop == 'iter' and self.analysis_to_obs:
            for isamp in range(self.nsamples):
                dir = os.path.join(self._sample_dir(isamp),
                                   f'{self.da_iteration + 1:d}')
                shutil.rmtree(dir)

    # internal methods
    def _sample_dir(self, isample):
        # print('in _sample_dir')
        "Return name of the sample's directory. "
        return os.path.join(self.results_dir, f'sample_{isample:d}')

    def _modify_foam_case(self, g, step, foam_dir=None):
        print('_modify_foam_case')
        for i, g_data in enumerate(self.g_data_list):
            g_data['internal_field']['value'] = g[:, i]
            if foam_dir is not None:
                g_data['file'] = os.path.join(foam_dir, str(step), f'g{i+1}')
            _ = rf.foam.write_field_file(**g_data)


def _run_foam(foam_dir, iteration, timeprecision):
    print('_run_foam')

    # run foam
    os.chdir(foam_dir)

    bash_command = "./run.sh"
    subprocess.call(bash_command, shell=True)

    os.chdir('../../')
    print('sleeping')
    time.sleep(10)
    print('end sleeping')
