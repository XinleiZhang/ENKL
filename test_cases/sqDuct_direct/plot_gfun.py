#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import pdb

# import foam_utilities as foam
from dafi.random_field import foam

if os.path.isdir('./results_figures'):
   None
else:
   os.mkdir('./results_figures')

# set plot properties
params = {
    # 'text.latex.preamble': ['\\usepackage{gensymb}'],
    # 'image.origin': 'lower',
    # 'image.interpolation': 'nearest',
    # 'image.cmap': 'viridis',
    'axes.grid': False,
    'savefig.dpi': 300, # 150,
    'axes.labelsize': 25, #13,
    'axes.titlesize': 25, # 10,
    'font.size': 25, #10,
    'legend.fontsize': 25,# 10,
    'xtick.labelsize': 25, #8,
    'ytick.labelsize': 25, #8,
    # 'text.usetex': True,
    'figure.figsize': [6,4],#[10,5],
    'font.family': 'serif',#'normal', #
    # 'font.weight': 'normal',
}
mpl.rcParams.update(params)

def shih_quadratic(theta):
    def g1(theta):
        num = -2./3.
        denom = 1.25 + np.sqrt(2 * theta[:, 0]) + 0.9 * np.sqrt(-2 * theta[:, 1])
        return num/denom

    def g234(theta, coeff):
        return coeff / (1000. + (2 * theta[:, 0])**(3./2.))

    g = np.empty([len(theta), 4])
    g[:, 0] = g1(theta)
    for i, c in enumerate([7.5, 1.5, -9.5]):
        g[:, i+1] = g234(theta, c)
    return g

Ub = 15.56
H = 2.0
Re = 5600.0
nu = Ub*H/Re

nsamps = 30
da_step = 99
plot_samps = 1
plot_median = False
plot_obs = 0
mesh_dir = './nnfoam_inputs/mesh'
base_dir = './nnfoam_inputs/foam_base'
truth_dir = './nnfoam_inputs/k_omega'
samps_dir = './results_ensemble'
output_dir = './results_figures'

# y = foam.read_cell_coordinates(mesh_dir)[:, 1] * 180
data_dir = 'nnfoam_inputs/data'
theta = np.load(os.path.join(data_dir, 'scalar_invariants.npy'))[:, :4]

obs = shih_quadratic(theta)[:,0]

Hx = np.loadtxt('results_dafi/t_0/Hx/Hx_{}'.format(da_step)).mean(axis=1)
fig, ax = plt.subplots()
ax.plot(theta[:,0], obs, 'k*', lw=3, markersize=1.5, label='Truth')
ax.plot(theta[:,0], -0.09 * np.ones(len(theta[:,0])), 'b*', lw=1.5, markersize=3,label='Prior')
ax.plot(theta[:,0], Hx, 'r*', lw=3, markersize=1.5, label='Posterior')
ax.set_ylabel(r'$g_1$')
ax.set_xlabel(r'$\theta$')
ax.tick_params(direction='in', length=8, width=2, colors='k',
           grid_color='k', grid_alpha=0.5)
plt.tight_layout()
plt.legend(fontsize = 12)
plt.show()
plt.close()

