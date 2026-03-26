#!/usr/bin/env python3

"""Copy and simplify code form original repo."""

import numpy as np
from tqdm import tqdm, trange

####################################
# FROM simulator.py
####################################


def correlated_poisson_spike_train(
    num_neurons: int,
    firing_rate: float,
    correlation: int,
    simulation_time: float,
    timestep: float,
    seed: int = 42,
):
    """Produces a set of Poisson process sampled spikes with a thinning process to achieve correlation

    This function creates the desired firing rate as a threshold and draws random numbers (tested against this threshold) to determine spikes.
    This is augmented with a shared random process to determine when neurons should share their activity with some global correlated spike-set.
    Based on Single Interaction Process Model described:
    Kuhn, A., Aertsen, A., & Rotter, S. (2003). Higher-order statistics of input ensembles and the response of simple model neurons. Neural Computation, 15(1), 67–101. https://doi.org/10.1162/089976603321043702

    Args:
        num_neurons (int): the number of neurons to simulate
        firing_rate (float): the firing rate to simulate for these neurons (spikes/ms)
        correlation (int): the within-group correlation
        simulation_time (float): the number of ms of simulation time (ms)
        timestep (float): the simulation timestepping (ms)

    Returns:
        spike_trains (list, np arrays): spike times over the simulation time per neuron (ms)
    """
    nb_timesteps = int(simulation_time / timestep)
    firing_rate_adjusted = firing_rate * timestep  # Converting to spikes/timestep

    # Creating a global spike train which all other spikes will correlate
    r = np.random.RandomState(seed)
    global_correlating_spiketrain = r.rand(nb_timesteps) < firing_rate_adjusted

    spike_trains = list()
    for n_indx in range(num_neurons):
        r = np.random.RandomState(seed + 2 + n_indx)

        neuron_spiketrain = (
            r.rand(nb_timesteps) < (1 - correlation) * firing_rate_adjusted
        )
        correlate_steps = r.rand(nb_timesteps) < correlation
        neuron_spiketrain[correlate_steps & global_correlating_spiketrain] = 1

        spike_trains.append(np.where(neuron_spiketrain)[0])
    return spike_trains


def random_sample_spike_train(
    spike_trains, simulation_time, timestep, resample_period, ratio_active, seed=42
):
    """Randomly samples units from a spike train to be active/inactive. This shifts all spike trains when inactive.

    Note, this function expects spike trains to have positive only values.

    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        simulation_time (int): maximum time of spikes (ms)
        timestep (float): time per step (ms)
        resample_period (int): time after which active/inactive neurons should be resampled
        ratio_active (float): the ratio of neurons to keep active during each period

    Returns:
        clipped_spike_trains (list, np arrays): spike trains within the min/max time only
    """
    nb_resamples = int(simulation_time // resample_period)
    nb_neurons = len(spike_trains)
    resample_period = int(resample_period / timestep)
    nb_on_units = int(ratio_active * nb_neurons)

    if nb_on_units < nb_neurons:
        r = np.random.RandomState(seed)
        on_unit_onehot = np.zeros((nb_neurons, nb_resamples))
        on_unit_onehot[:nb_on_units, :] = 1
        on_unit_onehot = np.asarray(
            [r.permutation(on_unit_onehot[:, x]) for x in range(nb_resamples)]
        ).transpose()
        on_segments = np.where(on_unit_onehot)
        for n_indx, s_indx in zip(on_segments[0], on_segments[1]):
            mask = (spike_trains[n_indx] > (resample_period * s_indx)) & (
                spike_trains[n_indx] <= (resample_period * (s_indx + 1))
            )
            spike_trains[n_indx][mask] *= -1

        for n_indx in range(nb_neurons):
            spike_trains[n_indx] = -spike_trains[n_indx][spike_trains[n_indx] < 0.0]

    return spike_trains


def xpsp_filterer(train, nb_timesteps, timestep, tau_slow, tau_fast):
    """Convolves a spike train with a double exponential causal XPSP filter

    Args:
        train (np array): list of spike times forming a single train
        nb_timesteps (int): the total number of timesteps of simulation
        timestep (float): timestep in ms
        tau_slow (float): slow decay time constant
        tau_fast (float): fast decay time constant

    Returns:
        xpsps (1D numpy array): the post synaptic potential for each timesteps
    """
    xpsp = np.zeros((nb_timesteps))
    xpsp[train] += 1.0 / (tau_slow - tau_fast)

    window_size = int(7 * (tau_slow / timestep))
    window_vals = timestep * np.arange(window_size)

    fast_filter = np.exp(-window_vals / tau_fast)
    fast_xpsp = np.convolve(xpsp, fast_filter)[: -(window_size - 1)]

    slow_filter = np.exp(-window_vals / tau_slow)
    slow_xpsp = np.convolve(xpsp, slow_filter)[: -(window_size - 1)]

    return slow_xpsp - fast_xpsp


def spike_trains_to_xpsps(
    spike_trains, sim_time, timestep, tau_slow=10.0, tau_fast=3.0
):
    """Converts a list of spike trains into a 2D numpy array of post-synaptic potentials

    Assumes that all spikes cause an equivalent shaped fast E/I PSP -- all psps are positive

    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        sim_time (float): total simulation time in ms
        timestep (float): timestep in ms
        tau_slow (float, optional): slow decay time constant
        tau_fast (float, optional): fast decay time constant

    Returns:
        xpsps (2D numpy array): the post synaptic potentials for all timesteps per neuron
    """
    nb_timesteps = int(sim_time / timestep)
    xpsps = np.zeros((len(spike_trains), nb_timesteps), dtype=np.float32)
    for n_indx, train in tqdm(
        enumerate(spike_trains), desc="spike trains to xpsps, nth spike train"
    ):
        xpsps[n_indx] = xpsp_filterer(train, nb_timesteps, timestep, tau_slow, tau_fast)
    return xpsps


def lif_dynamics(
    xpsps,
    weight_matrix,
    timestep,
    tau=20.0,
    thresh=1.0,
    rest=0.0,
    reset=-1.0,
    drift=0.0,
    coupling_ratio=1.0,
):
    """Computing leaky integrator spiking neuron dynamics given an incident XPSP and weight matrix

    Args:
        xpsps (2D numpy array, float): post synaptic potentials by pre-synaptic neuron population
        weight_matrix (2D np array, float): a weight matrix of size (postsynaptic population, presynaptic population)
        timestep (float): the timestep of simulated dynamics (Forward-Euler method)
        tau (float, optional): the time constant of the leaky integrator
        thresh (float, optional): the threshold which determines when a neuron spikes
        rest (float, optional): the resting membrane voltage, baseline
        reset (float, optional): the voltage to which neuron membranes are reset after a spike
        drift (float, optional): the constant background input to a cells
        coupling_ratio (float, optional): the dendritic vs somatic coupling ratio (g_D / g_L)

    Returns:
        DEPRECATED acc_voltage (2D np array, float): the membrane voltage of output population without resetting at spike times
        DEPRECATED mem_voltage (2D np array, float): the membrane voltages of output population for the duration
        spike_times (list of numpy arrays, float): spike times of the output population
        NOTE: acc_voltagea and mem_voltage probably not needed in STDWI
        NOTE: only for RDD --> don't store it?
    """
    nb_post_neurons = weight_matrix.shape[0]
    nb_timesteps = xpsps.shape[1]
    mem_voltage = np.zeros(nb_post_neurons, dtype=np.float32)
    acc_voltage = np.zeros(mem_voltage.shape, dtype=np.float32)
    spike_times = [[] for p in range(nb_post_neurons)]
    timestep_inputs = np.einsum("ij, jn->in", weight_matrix, xpsps)

    for t_indx in trange(1, nb_timesteps, desc="lif dynamics, time step"):
        total_input = timestep_inputs[:, t_indx - 1]
        # Membrane voltage update
        dmem = ((rest + drift) - mem_voltage) + coupling_ratio * (
            total_input - mem_voltage
        )
        mem_voltage = mem_voltage + (timestep / tau) * dmem
        dacc = ((rest + drift) - acc_voltage) + coupling_ratio * (
            total_input - acc_voltage
        )
        acc_voltage = acc_voltage + (timestep / tau) * dacc
        # Spike reset
        mask = mem_voltage >= thresh
        mem_voltage[mask] = reset
        # Storing spikes
        spiked_neurons = np.where(mask)[0]
        for s in spiked_neurons:
            spike_times[s].append(t_indx)

    for n in range(nb_post_neurons):
        spike_times[n] = np.asarray(spike_times[n]).astype(int)
    return spike_times


def binary_spike_matrix(spike_trains, sim_time, timestep):
    """Converts a list of spike trains into a large NxT binary spike matrix

    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        sim_time (float): simulation time total
        timestep (float): timestep with which to divide the simulation time

    Returns:
        binary_spike_matrix (2D array)
    """
    nb_timesteps = int(sim_time / timestep)
    nb_neurons = len(spike_trains)

    binary_matrix = np.zeros((nb_neurons, nb_timesteps), dtype=np.float32)
    for n_indx in range(nb_neurons):
        spiketime_indices = (spike_trains[n_indx] / timestep).astype(int)
        binary_matrix[n_indx, spiketime_indices] = 1.0
    return binary_matrix


####################################
# FROM methods.py
####################################


def stdwi_method(
    guess_matrix,
    input_binary_spikes,
    output_binary_spikes,
    slow_in_trace,
    fast_in_trace,
    learning_rate,
    decay_weighting,
):
    """Spike Timing-Dependent based inference of weights

    Args:
        guess_matrix (IxJ float array): A prior estimate of the weight matrix
        input_binary_spikes (JxN binary array): A binary matrix giving spike location of M input neuron in N timesteps
        output_binary_spikes (IxN binary array): A binary matrix giving spike location of M output neuron in N timesteps
        slow_in_trace (JxN float array): A slow exponential moving trace of the input neuron spikes for the N timesteps
        fast_in_trace (JxN float array): A fast exponential moving trace of the input neuron spikes for the N timesteps
        learning_rate (float): A weighting of the synaptic inference update
        decay_weighting (float): The relative strength of depression

    Returns:
        update_matrix (IxJ float array): the measured updated to the guess matrix given the data
    """
    update_matrix = np.copy(guess_matrix)
    nb_sub_timesteps = input_binary_spikes.shape[1]
    relative_input_trace = fast_in_trace - slow_in_trace

    for t_indx in range(nb_sub_timesteps):
        # Potentiation is based upon output spike times also but also by mult. with the relevant input traces
        LTP_update = np.matmul(
            output_binary_spikes[:, t_indx][:, np.newaxis],
            relative_input_trace[:, t_indx][np.newaxis, :],
        )

        # Calculating depression based upon output spikes
        LTD_update = np.copy(update_matrix)
        LTD_mask = np.repeat(
            output_binary_spikes[:, t_indx][:, np.newaxis],
            update_matrix.shape[1],
            axis=1,
        )
        LTD_update = LTD_update * LTD_mask
        # Complete update based upon LTP and LTD incl. learning rate
        update = learning_rate * (LTP_update - decay_weighting * LTD_update)
        update_matrix = update_matrix + update

    return update_matrix


def create_stdwi_trace(prev_trace, binary_spike_matrix, alpha, tau, timestep, alltoall):
    """Produces an exponential moving average estimation of firing rate from spike times

    Args:
        prev_trace (MxN float matrix): The trace which is being continued (normally initialised with zeros)
        binary_spike_matrix (MxN int matrix): A binary matrix indicating spike times of the M neurons in N timesteps
        alpha (float): The height of exponential moving average filter
        tau (float): Time constant of the exponential moving average filter
        timestep (float): Timestep of spiking data to compute traces with tau
        alltoall (bool): All to all vs. nearest only trace
    Returns:
        next_trace (MxN float matrix): The trace given the prev_trace and provided spikes/config
    """
    next_trace = np.zeros(prev_trace.shape, dtype=np.float32)
    decay_factor = np.exp(-timestep / tau)
    nb_sub_timesteps = prev_trace.shape[1]

    next_trace[:, 0] = decay_factor * prev_trace[:, -1]
    if alltoall:
        next_trace += alpha * binary_spike_matrix

    for t_indx in np.arange(1, nb_sub_timesteps):
        next_trace[:, t_indx] += decay_factor * next_trace[:, t_indx - 1]
        if not alltoall:
            next_trace[binary_spike_matrix[:, t_indx] > 0.0, t_indx] = alpha

    return next_trace


def apply_stdwi(
    inp_spks,
    out_spks,
    fb_weight,
    tau_fast,
    tau_slow,
    sim_dur,
    stim_dur,
    lr,
    dt,
    decay_weighting=0.1,
    a_fast=1.0,
):
    """My own leaner reimplementation of fitter.stdwi"""

    n_in = fb_weight.shape[1]
    n_sim_timesteps = int(sim_dur / dt)
    n_stims = int(sim_dur / stim_dur)

    a_slow = a_fast * tau_fast / tau_slow

    # NOTE: from here on, the spikes will be converted into units of ms!
    inp_spks = [spks * dt for spks in inp_spks]
    out_spks = [spks * dt for spks in out_spks]

    inp_spks_bin = binary_spike_matrix(inp_spks, sim_dur, dt)
    out_spks_bin = binary_spike_matrix(out_spks, sim_dur, dt)

    slow_input_trace = np.zeros((n_in, n_sim_timesteps), dtype=np.float32)
    fast_input_trace = np.zeros_like(slow_input_trace)

    slow_input_trace = create_stdwi_trace(
        slow_input_trace, inp_spks_bin, a_slow, tau_slow, dt, alltoall=True
    )
    fast_input_trace = create_stdwi_trace(
        fast_input_trace, inp_spks_bin, a_fast, tau_fast, dt, alltoall=True
    )

    # iterate over the stimulation periods
    len_stim = int(stim_dur / dt)
    for i in trange(n_stims, desc="stdwi stimulation periods"):
        # get trace chunks for the stim perdiod
        i_start = i * len_stim
        i_end = (i + 1) * len_stim
        chunk_slice = np.s_[
            :,
            i_start:i_end,
        ]
        fb_weight = stdwi_method(
            fb_weight,
            inp_spks_bin[chunk_slice],
            out_spks_bin[chunk_slice],
            slow_input_trace[chunk_slice],
            fast_input_trace[chunk_slice],
            lr,
            decay_weighting=decay_weighting,
        )

    return fb_weight
