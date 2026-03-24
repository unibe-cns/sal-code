#!/usr/bin/env python3


import numpy as np

# ---------------------------
# Define some (shared) parameters mainly (or exclusively?) used by RDDNet
# ---------------------------

dt = 0.001  # timestep (s)
v_reset = -1  # reset somatic voltage
spike_threshold = 1  # spike threshold
refractory_time = np.ceil(0.003 / dt)  # refractory time (timesteps)
u_window = 10  # input drive window used to determine near-threshold data points for RDD
tau_s = 0.003  # synaptic time constant (s) - used to calculate input
tau_L = 0.01  # leak time constant (s) - used to calculate input
mem = 30  # spike history memory length (timesteps)
RDD_window = 0.03 / dt  # RDD integration window length (timesteps)
RDD_init_window = 0.1  # window around spike threshold that determines when an RDD integration window is initiated
input_rate = 200 * dt  # input spike rate
alpha = 4  # driving input weight
g_L = 0.1 / dt  # leak conductance
g_D = 0.6 / dt  # dendritic conductance
RDD_eta = 0.001  # RDD learning rate


# ---------------------------
# Define the network sizes of the common classification layers
# ---------------------------

CLASSIFICATION_LAYER_SIZES = [1600, 384, 192, 10]


# ---------------------------
# Define import structure of modules
# ---------------------------

from .conv_net import ConvNet  # noqa
from .rdd_net import RDDNet  # noqa
