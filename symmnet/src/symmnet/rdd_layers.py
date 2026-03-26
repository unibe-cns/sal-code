"""Implementation of the dynamics of RDD layers."""

import numpy as np
from symmnet import (
    RDD_eta,
    RDD_init_window,
    RDD_window,
    alpha,
    dt,
    g_D,
    g_L,
    mem,
    refractory_time,
    spike_threshold,
    tau_L,
    tau_s,
    u_window,
    v_reset,
)


def kappa(x):
    """
    Computes the difference of exponentials kernel for synaptic current.

    Args:
        x (float or int): Time step index.

    Returns:
        float: Value of the kernel at time x.
    """
    return (np.exp(-x / (tau_L / dt)) - np.exp(-x / (tau_s / dt))) / (
        (tau_L / dt) - (tau_s / dt)
    )


def get_kappas(n=mem):
    """
    Computes the kappa kernel for the last n time steps.

    Args:
        n (int): Number of memory steps.

    Returns:
        np.ndarray: Array of kappa values for each time step.
    """
    return np.array([kappa(i + 1) for i in range(n)])


# Precompute and flip the kappa kernel for convolution with recent history.
kappas = np.flipud(get_kappas(mem))[:, np.newaxis]  # initialize kappas array


class SpikingFA:
    """
    Implements a population of spiking neurons with optional
    Regression Discontinuity Design (RDD) for causal inference of feedback weights.

    If b_input_size is provided, RDD logic is enabled for the neuron population.

    Args:
        size (int): Number of neurons in the population.
        f_input_size (int, optional): Size of the feedforward input.
        b_input_size (int, optional): Size of the feedback input (enables RDD).
    """

    def __init__(self, size, f_input_size=None, b_input_size=None):
        self.size = size
        self.f_input_size = f_input_size
        self.b_input_size = b_input_size

        if self.f_input_size is not None:
            self.weight = np.zeros((self.size, self.f_input_size))

        self.RDD = self.b_input_size is not None

        self.weight = None
        self.fb_weight = None
        self.fb_weight_std = None

        if self.RDD:
            # RDD_params shape: (neurons, 4, feedback_inputs)
            # Stores regression coefficients for causal estimation at the threshold
            self.RDD_params = np.zeros((self.size, 4, self.b_input_size))
            self.beta = np.zeros((self.size, self.b_input_size))

        self.reset()

    def reset(self):
        """
        Resets all state variables for the spiking neuron population.
        """
        self.v = v_reset * np.ones((self.size, 1))
        self.B = np.zeros((self.size, 1))

        self.fired = np.zeros((self.size, 1), dtype=bool)  # spike events ?
        self.refractory_time_left = np.zeros((self.size, 1), dtype=int)
        self.spike_hist = np.zeros((self.size, mem), dtype=int)

        if self.RDD:
            self.fb_weight = np.zeros((self.size, self.b_input_size))
            self.fb_weight_orig = np.zeros((self.size, self.b_input_size))

            self.u = v_reset * np.ones((self.size, 1))

            self.RDD_time_left = np.zeros((self.size, 1))
            self.R = np.zeros((self.size, self.b_input_size))
            self.R_pre = np.zeros((self.size, self.b_input_size))
            self.n_spikes = np.zeros((self.size, 1))
            self.max_u = np.zeros((self.size, 1))

    def set_weights(self, weight=None, bias=None, fb_weight=None):
        """
        Sets the feedforward and feedback weights, with normalization.

        Args:
            weight (np.ndarray, optional): Feedforward weights.
            bias (np.ndarray, optional): Not used, kept for compatibility.
            fb_weight (np.ndarray, optional): Feedback weights.
        """
        if weight is not None:
            self.weight_orig = weight.copy()
            self.weight = 0.2 * self.f_input_size * weight / np.std(weight)

        if fb_weight is not None:
            self.fb_weight = fb_weight.copy()

            if self.fb_weight_std is None:
                self.fb_weight_std = np.std(self.fb_weight)

    def update(self, f_input=None, b_input=None, driving_input=None):
        """
        Advances the state of the neuron population by one time step.

        This includes updating membrane potentials, spike histories,
        and, if enabled, running the RDD logic.

        Args:
            f_input (np.ndarray, optional): Feedforward input.
            b_input (np.ndarray, optional): Feedback input (used for RDD).
            driving_input (np.ndarray, optional): External driving input.
        """
        if self.RDD:
            # determine which neurons are just ending their RDD integration window
            self.RDD_window_ending_mask = self.RDD_time_left == 1

            # update maximum input drives for units in their RDD integration window
            self.max_u[np.logical_and(self.RDD_time_left > 0, self.u > self.max_u)] = (
                self.u[np.logical_and(self.RDD_time_left > 0, self.u > self.max_u)]
            )

        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.refractory_time_left > 0] -= 1

        # calculate basal potential
        if driving_input is not None:
            p = np.dot(driving_input, kappas)
            self.B = alpha * p
        elif f_input is not None:
            p = np.dot(f_input, kappas)
            self.B = np.dot(self.weight, p)
        else:
            self.B *= 0

        if self.RDD:
            # calculate apical potential
            q = np.dot(b_input, kappas)

        # calculate changes in voltages and input drives, and update both
        self.dv_dt = -g_L * self.v + g_D * (self.B - self.v)
        self.v += dt * self.dv_dt

        if self.RDD:
            self.du_dt = -g_L * self.u + g_D * (self.B - self.u)
            self.u += dt * self.du_dt

        if self.RDD:
            # update rewards for units in their RDD integration window
            self.R[np.squeeze(self.RDD_time_left > 0)] += q[:, 0]
            self.R_pre[np.squeeze(self.RDD_time_left == 0)] = q[:, 0]

        # determine which neurons are in a refractory period
        refractory_mask = self.refractory_time_left > 0

        # determine which neurons are above spiking threshold
        threshold_mask = self.v >= spike_threshold

        # neurons above threshold that are not in their refractory period will spike
        self.fired *= False
        self.fired[
            np.logical_and(
                threshold_mask, refractory_mask == False  # noqa  (E712 error)
            )
        ] = True

        # reset voltages of neurons that spiked
        self.v[self.fired] = v_reset

        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.fired] = refractory_time

        if self.RDD:
            if driving_input is not None:
                # update RDD estimates (only neurons whose RDD integration window has ended will update their estimate)
                self.update_RDD_estimate()

            # decrement time left in RDD integration windows
            self.RDD_time_left[self.RDD_time_left > 0] -= 1

            # reset the input drive to match the voltage, for neurons that are not in an RDD integration window
            self.u[self.RDD_time_left == 0] = self.v[self.RDD_time_left == 0]

        if self.RDD:
            # determine which neurons are starting a new RDD integration window
            self.new_RDD_window_mask = np.logical_and(
                np.abs(spike_threshold - self.u) <= RDD_init_window,
                self.RDD_time_left == 0,
            )

            self.RDD_time_left[self.new_RDD_window_mask] = RDD_window

            # update number of spikes that have occurred during RDD integration windows
            self.n_spikes[np.logical_and(self.RDD_time_left > 0, self.fired)] += 1

            # reset RDD variables for neurons whose RDD integration window has ended
            self.n_spikes[self.RDD_window_ending_mask] = 0
            self.R[np.squeeze(self.RDD_window_ending_mask)] = 0
            self.max_u[self.RDD_window_ending_mask] = 0

        # update spike histories
        self.spike_hist = np.concatenate([self.spike_hist[:, 1:], self.fired], axis=1)

    def update_RDD_estimate(self):
        """
        Updates the RDD regression parameter estimates for neurons at the end of their RDD window.
        """
        # figure out which neurons are at the end of their RDD integration window, and either just spiked or almost spiked
        just_spiked_mask = np.logical_and(
            self.RDD_window_ending_mask,
            np.logical_and(
                np.abs(self.max_u - spike_threshold) <= u_window, self.n_spikes >= 1
            ),
        )[:, 0]
        almost_spiked_mask = np.logical_and(
            self.RDD_window_ending_mask,
            np.logical_and(
                np.abs(self.max_u - spike_threshold) <= u_window, self.n_spikes < 1
            ),
        )[:, 0]

        # update RDD estimates for neurons that just spiked or almost spiked
        if np.sum(just_spiked_mask) > 0:
            self.R[just_spiked_mask] /= RDD_window
            err = (
                self.RDD_params[just_spiked_mask, 2] * self.max_u[just_spiked_mask]
                + self.RDD_params[just_spiked_mask, 0]
                - (self.R[just_spiked_mask] - self.R_pre[just_spiked_mask])
            )

            self.RDD_params[just_spiked_mask, 2] -= (
                RDD_eta * err * self.max_u[just_spiked_mask]
            )
            self.RDD_params[just_spiked_mask, 0] -= RDD_eta * err
        if np.sum(almost_spiked_mask) > 0:
            self.R[almost_spiked_mask] /= RDD_window
            err = (
                self.RDD_params[almost_spiked_mask, 3] * self.max_u[almost_spiked_mask]
                + self.RDD_params[almost_spiked_mask, 1]
                - (self.R[almost_spiked_mask] - self.R_pre[almost_spiked_mask])
            )

            self.RDD_params[almost_spiked_mask, 3] -= (
                RDD_eta * err * self.max_u[almost_spiked_mask]
            )
            self.RDD_params[almost_spiked_mask, 1] -= RDD_eta * err

        end_mask = np.logical_or(just_spiked_mask, almost_spiked_mask)

        self.beta[end_mask] = (
            self.RDD_params[end_mask, 2] * spike_threshold
            + self.RDD_params[end_mask, 0]
            - (
                self.RDD_params[end_mask, 3] * spike_threshold
                + self.RDD_params[end_mask, 1]
            )
        )

        self.R[end_mask] = 0

    def update_fb_weights(self):
        """
        Updates the feedback weights based on the current RDD beta values, if nonzero.
        """
        mask = self.beta != 0

        if np.sum(mask) > 0:
            self.fb_weight[mask] = (
                self.beta[mask] * self.fb_weight_std / np.std(self.beta[mask])
            )
