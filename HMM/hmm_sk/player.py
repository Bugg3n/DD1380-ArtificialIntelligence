#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
import sys


epsilon = sys.float_info.epsilon

def forward_backward(transition_matrix, observation_matrix, initial_probabilities, observation_sequence):
    num_states = len(transition_matrix)
    sequence_length = len(observation_sequence)

    iteration_count = 0
    max_iterations = 5
    old_log_probability = float("-inf")
    log_probability = 1

    while iteration_count < max_iterations and log_probability > old_log_probability:
        iteration_count += 1
        if iteration_count != 1:
            old_log_probability = log_probability
            
        alpha_values, scaling_factors = forward_pass(transition_matrix, observation_matrix, initial_probabilities, observation_sequence, num_states, sequence_length)
        
        # Reverse the observation sequence for the backward pass
        reversed_observation_sequence = observation_sequence[::-1]
        reversed_scaling_factors = scaling_factors[::-1]

        beta_values_reversed = backward_pass(transition_matrix, observation_matrix, initial_probabilities, reversed_observation_sequence, reversed_scaling_factors, num_states, sequence_length)
        beta_values = beta_values_reversed[::-1]

        gamma_values, gamma_ij_values = compute_gamma(transition_matrix, observation_matrix, observation_sequence, alpha_values, beta_values, num_states, sequence_length)

        initial_probabilities, transition_matrix, observation_matrix = re_estimate(gamma_values, gamma_ij_values, observation_sequence, sequence_length, num_states)

        log_probability = calculate_log_probability(scaling_factors, sequence_length) 

    return transition_matrix, observation_matrix, initial_probabilities



def create_matrix(matrix_items):
    num_rows = int(matrix_items[0])
    num_cols = int(matrix_items[1])
    matrix_elements = list(map(float, matrix_items[2:]))

    matrix = []
    for row in range(num_rows):
        start_index = row * num_cols
        end_index = start_index + num_cols
        matrix_row = matrix_elements[start_index:end_index]
        matrix.append(matrix_row)

    return matrix


def forward_pass(transition_matrix, observation_matrix, initial_probabilities, observation_sequence, num_states, sequence_length):
    alpha_values = []
    scaling_factors = []
    
    for time_step in range(sequence_length):
        current_alpha = []
        scaling_factor = 0

        for state in range(num_states):
            if time_step == 0:
                alpha = initial_probabilities[0][state] * observation_matrix[state][observation_sequence[time_step]]
            else:
                alpha = sum(alpha_values[time_step - 1][previous_state] * transition_matrix[previous_state][state] 
                            for previous_state in range(num_states)) * observation_matrix[state][observation_sequence[time_step]]
            
            scaling_factor += alpha
            current_alpha.append(alpha)

        # Avoid division by zero
        scaling_factor_reciprocal = 1 / (scaling_factor + sys.float_info.epsilon)
        current_alpha = [alpha * scaling_factor_reciprocal for alpha in current_alpha]

        scaling_factors.append(scaling_factor_reciprocal)
        alpha_values.append(current_alpha)

    return alpha_values, scaling_factors



def backward_pass(transition_matrix, observation_matrix, initial_probabilities, observation_sequence, scaling_factors, num_states, sequence_length):
    beta_values = [None] * sequence_length

    # Initialize the last time step with scaled values
    beta_values[-1] = [scaling_factors[-1]] * num_states

    # Iterate backwards through the sequence
    for time_step in range(sequence_length - 2, -1, -1):
        beta_values[time_step] = []
        for i in range(num_states):
            beta = sum(beta_values[time_step + 1][j] * transition_matrix[i][j] * observation_matrix[j][observation_sequence[time_step]]
                       for j in range(num_states))
            # Scale the beta values
            beta_values[time_step].append(beta * scaling_factors[time_step])

    return beta_values

    
    
def compute_gamma(transition_matrix, observation_matrix, observation_sequence, alpha_values, beta_values, num_states, sequence_length):
    gamma_values = []
    gamma_ij_values = []

    for time_step in range(sequence_length - 1):
        gamma_at_t = []
        gamma_ij_at_t = []

        for i in range(num_states):
            gamma_sum = 0
            gamma_ij_for_state = []

            for j in range(num_states):
                gamma_ij = alpha_values[time_step][i] * transition_matrix[i][j] * observation_matrix[j][observation_sequence[time_step + 1]] * beta_values[time_step + 1][j]
                gamma_sum += gamma_ij
                gamma_ij_for_state.append(gamma_ij)

            gamma_at_t.append(gamma_sum)
            gamma_ij_at_t.append(gamma_ij_for_state)

        gamma_values.append(gamma_at_t)
        gamma_ij_values.append(gamma_ij_at_t)

    # Handling the last time step for gamma
    gamma_last = [alpha_values[-1][state] for state in range(num_states)]
    gamma_values.append(gamma_last)

    return gamma_values, gamma_ij_values



def re_estimate(gamma_values, gamma_ij_values, observation_sequence, num_observations, num_states, sequence_length):
    # Re-estimate initial probabilities (pi)
    initial_probabilities = [gamma_values[0][state] for state in range(num_states)]

    # Re-estimate transition matrix (A)
    new_transition_matrix = []
    for i in range(num_states):
        denominator = sum(gamma_values[t][i] for t in range(sequence_length - 1))
        new_transition_row = [
            sum(gamma_ij_values[t][i][j] for t in range(sequence_length - 1)) / (denominator + sys.float_info.epsilon)
            for j in range(num_states)
        ]
        new_transition_matrix.append(new_transition_row)

    # Re-estimate observation matrix (B)
    new_observation_matrix = []
    for i in range(num_states):
        denominator = sum(gamma_values[t][i] for t in range(sequence_length))
        new_observation_row = [
            sum(gamma_values[t][i] for t in range(sequence_length) if observation_sequence[t] == j) / (denominator + sys.float_info.epsilon)
            for j in range(num_observations)
        ]
        new_observation_matrix.append(new_observation_row)

    return [initial_probabilities], new_transition_matrix, new_observation_matrix

def forward_algorithm(observations, model):
    observation_matrix_transposed = transpose(model.B)
    alpha = dot_product(model.PI, observation_matrix_transposed[observations[0]])

    for observation in observations[1:]:
        alpha = dot_product(matrix_multiplication(alpha, model.A), observation_matrix_transposed[observation])

    return sum(alpha[0])

def calculate_log_probability(scaling_factors, sequence_length):
    log_probability = -sum(np.log(scaling_factors[t]) for t in range(sequence_length))
    return log_probability

def dot_product(matrix_a, vector_b):
    return [sum(a * b for a, b in zip(matrix_a_row, vector_b)) for matrix_a_row in matrix_a]

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def matrix_multiplication(matrix_a, matrix_b):
    return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*matrix_b)] for row_a in matrix_a]

def gen_stochastic_row(size):
    random_values = np.random.rand(size) / 1000
    uniform_values = np.full(size, 1 / size)
    stochastic_row = uniform_values + random_values
    return stochastic_row / stochastic_row.sum()

class Model:
    def __init__(self, num_states, num_emissions):
        """
        Initializes the Model with given number of states and emissions.
        
        Args:
        num_states (int): The number of states in the HMM.
        num_emissions (int): The number of possible emission symbols.
        """
        self.PI = gen_stochastic_row(num_states)
        self.A = [gen_stochastic_row(num_states) for _ in range(num_states)]
        self.B = [gen_stochastic_row(num_emissions) for _ in range(num_states)]

    @property
    def transition_matrix(self):
        return self.A

    @transition_matrix.setter
    def transition_matrix(self, matrix):
        self.A = matrix

    @property
    def observation_matrix(self):
        return self.B

    @observation_matrix.setter
    def observation_matrix(self, matrix):
        self.B = matrix

    @property
    def initial_probabilities(self):
        return self.PI

    @initial_probabilities.setter
    def initial_probabilities(self, probabilities):
        self.PI = probabilities

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def __init__(self):
        self.models_fish = [Model(1, N_EMISSIONS) for _ in range(N_SPECIES)]
        self.fishes = [(i, []) for i in range(N_FISH)]
        self.obs = None

    def update_model(self, model_id, true_type):
        """
        Update the HMM model for the given fish type.
        """
        model = self.models_fish[model_id]
        updated_A, updated_B, updated_PI = forward_backward(model.A, model.B, model.PI, self.obs)
        model.transition_matrix = updated_A
        model.observation_matrix = updated_B
        model.initial_probabilities = updated_PI

    def guess(self, step, observations):
        """
        Make a guess about the fish type based on the observations.
        """
        for i, observation in enumerate(observations):
            self.fishes[i][1].append(observation)

        if step < 110:  # Wait for enough observations
            return None

        fish_id, obs = self.fishes.pop(0)
        fish_type, max_prob = self.identify_fish_type(obs)

        self.obs = obs
        return fish_id, fish_type

    def identify_fish_type(self, observations):
        """
        Identify the fish type based on observations.
        """
        max_prob = 0
        identified_type = 0
        for model_id, model in enumerate(self.models_fish):
            probability = forward_algorithm(observations, model)
            if probability > max_prob:
                max_prob = probability
                identified_type = model_id
        return identified_type, max_prob

    def reveal(self, correct, fish_id, true_type):
        """
        Process the feedback received after making a guess.
        """
        if not correct:
            self.update_model(fish_id, true_type)
