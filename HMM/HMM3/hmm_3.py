import sys
import math

eps = sys.float_info.epsilon

# A = transition probability matrix
# B = observation probability matrix
# pi = initial state distribution
# emissions = emission matrix

def forward_with_scaling(A, B, pi, emissions):
    num_states = len(A)
    num_observations = len(B)

    # Initialize the scaling factor array
    scale_factors = [0.0] * num_observations

    # Initialize the forward matrix with zeros
    alpha = [[0.0] * num_states for _ in range(num_observations)]

    # Initialize the first column of the forward matrix
    for state in range(num_states):
        alpha[0][state] = pi[0][state] * emissions[state][B[0]]
        scale_factors[0] += alpha[0][state]

    # Scale the first column of the forward matrix
    scale_factors[0] = 1 / scale_factors[0]
    for state in range(num_states):
        alpha[0][state] *= scale_factors[0]

    # Fill in the rest of the forward matrix
    for t in range(1, num_observations):
        for j in range(num_states):
            alpha[t][j] = sum(alpha[t-1][i] * A[i][j] for i in range(num_states)) * emissions[j][B[t]]
            scale_factors[t] += alpha[t][j]

        # Scale the t-th column of the forward matrix
        scale_factors[t] = 1 / scale_factors[t]
        for i in range(num_states):
            alpha[t][i] *= scale_factors[t]

    return alpha, scale_factors

def normalize(matrix):
    for row in matrix:
        row_sum = sum(row)
        if row_sum > 0:
            row[:] = [x / row_sum for x in row]


def backward_with_scaling(A, B, emissions, scale_factors):
    num_states = len(A)
    num_observations = len(B)

    # Initialize the backward matrix with zeros
    backward_matrix = [[0.0] * num_states for _ in range(num_observations)]

    # Initialize the last row of the backward matrix
    for state in range(num_states):
        backward_matrix[-1][state] = scale_factors[-1]

    # Fill in the rest of the backward matrix
    for t in range(num_observations - 2, -1, -1):
        for i in range(num_states):
            backward_matrix[t][i] = sum(A[i][j] * emissions[j][B[t + 1]] * backward_matrix[t + 1][j] for j in range(num_states))
            backward_matrix[t][i] *= scale_factors[t]

    return backward_matrix

def baum_welch(A, B, pi, emissions, iterations=100, threshold=1e-6, old_log_prob = -math.inf):

    for _ in range(iterations):
        alpha, scale_factors = forward_with_scaling(A, B, pi, emissions)
        beta = backward_with_scaling(A, B, emissions, scale_factors)

        # E-step
        gamma, xi = calculate_gamma(alpha, beta, B, A, emissions)

        # M-step: Update A and B with change tracking
        pi_new, A_new, B_new = estimate(A, emissions, pi, B, gamma, xi)

        # Convergence Check
        log_prob = compute_log(scale_factors, B)
        if  log_prob > old_log_prob:
                old_log_prob = log_prob
        else:
            break

    return A_new, B_new

def estimate(A, emissions, pi, B, gamma, xi):
    num_states = len(A)
    num_observations = len(B)
    num_emissions = len(emissions[0])

    # Re-estimate initial state distribution
    for i in range(num_states):
        pi[0][i] = gamma[0][i]

    # Re-estimate transition matrix (A)
    for i in range(num_states):
        denom = sum(gamma[t][i] for t in range(num_observations - 1))
        for j in range(num_states):
            numer = sum(xi[t][i][j] for t in range(num_observations - 1))
            A[i][j] = numer / denom if denom != 0 else 0

    # Re-estimate emission matrix (B)
    for i in range(num_states):
        denom = sum(gamma[t][i] for t in range(num_observations))
        for j in range(num_emissions):
            numer = sum(gamma[t][i] for t in range(num_observations) if B[t] == j)
            emissions[i][j] = numer / denom if denom != 0 else 0

    return pi, A, emissions


def calculate_gamma(alpha, beta, observations, A, emissions):
    num_states = len(A)
    num_observations = len(observations)

    gamma = [[0.0] * num_states for _ in range(num_observations)]
    xi = [[[0.0 for _ in range(num_states)] for _ in range(num_states)] for _ in range(num_observations)]

    # Compute gamma and di_gamma for each time step
    for t in range(num_observations - 1):
        for i in range(num_states):
            total = 0.0
            for j in range(num_states):
                di_gamma_value = alpha[t][i] * A[i][j] * emissions[j][observations[t + 1]] * beta[t + 1][j]
                xi[t][i][j] = di_gamma_value
                total += di_gamma_value

            gamma[t][i] = total

    # Special case for the last time step
    for i in range(num_states):
        gamma[-1][i] = alpha[-1][i]

    return gamma, xi


def compute_log(c,obs):
    log_prob = 0
    num_obs = len(obs)
    for i in range(num_obs): 
        log_prob = log_prob + math.log(c[i])
    log_prob = -log_prob

    return log_prob


def read_matrix():
    parts = sys.stdin.readline().strip().split()
    rows, cols = int(parts[0]), int(parts[1])
    elements = list(map(float, parts[2:]))
    return [elements[i * cols:(i + 1) * cols] for i in range(rows)]


def format_output(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    formatted_output = f"{rows} {cols} " + " ".join(f"{num:.6f}" for row in matrix for num in row)
    return formatted_output

def parse_matrix(input_lines):
    tokens = [float(token) for token in input_lines.split()] # Split the input into tokens and convert to float
    rows_num, cols_num = int(tokens[0]), int(tokens[1]) # Extract the number of rows and columns
    return [tokens[2 + cols_num * row: 2 + cols_num * (row + 1)] for row in range(rows_num)] # Extract the elements and organize them into a matrix

def parse_observations(input_lines):
    tokens = input_lines.split() # Split the input into tokens
    obs_num = int(tokens[0]) # Extract the number of observations
    return [int(token) for token in tokens[1:1 + obs_num]] # Convert the remaining tokens to integers and create the observations list

def main():
    A= parse_matrix(input())
    emissions = parse_matrix(input())
    pi = parse_matrix(input())
    B = parse_observations(input())

    # Read the emission sequence
    # _, *emissions = list(map(int, sys.stdin.readline().strip().split()))

    # Run Baum-Welch algorithm until local maximum is reached
    updated_A, updated_B = baum_welch(A, B, pi, emissions)

    # Print updated matrices
    sys.stdout.write(format_output(updated_A) + '\n')
    sys.stdout.write(format_output(updated_B) + '\n')


if __name__ == "__main__":
    main()
