import sys


def format_input():
    """
    Reads input from the terminal and formats it into matrices and an emission sequence.
    """
    matrices = []
    for _ in range(3):  # Expecting three matrices
        line = sys.stdin.readline().strip()
        parts = line.split()
        rows, cols = int(parts[0]), int(parts[1])
        elements = parts[2:]
        matrix = [[float(elements[i * cols + j]) for j in range(cols)] for i in range(rows)]
        matrices.append(matrix)

    # Read the emission sequence
    emissions_line = sys.stdin.readline().strip()
    emissions_parts = emissions_line.split()
    emissions = [int(emissions_parts[i]) for i in range(1, len(emissions_parts))]

    return matrices, emissions


def viterbi_algorithm(A, B, initial_distribution, emissions):
    """
    Takes four matrixes and runs the viterbi algorithm.
    """
    num_states = len(A)
    num_observations = len(emissions)

    # Initialize delta and psi
    delta = [[0 for _ in range(num_states)] for _ in range(num_observations)]
    psi = [[0 for _ in range(num_states)] for _ in range(num_observations)]

    # Initialization step
    for i in range(num_states):
        delta[0][i] = initial_distribution[i] * B[i][emissions[0]]

    # Recursion step
    for t in range(1, num_observations):
        for j in range(num_states):
            (prob, state) = max(
                [(delta[t-1][i] * A[i][j] * B[j][emissions[t]], i) for i in range(num_states)]
            )
            delta[t][j] = prob
            psi[t][j] = state

    # Termination step
    last_state = delta[-1].index(max(delta[-1]))

    # Path backtracking
    path = [last_state]
    for t in range(num_observations - 1, 0, -1):
        path.insert(0, psi[t][path[0]])

    return path


def main():
    matrices, emissions = format_input()

    # Extract the individual matrices
    transition, emission, initial_state = matrices
    initial_distribution = initial_state[0]

    # Run Viterbi algorithm
    path = viterbi_algorithm(transition, emission, initial_distribution, emissions)

    # Output the most probable sequence of states
    sys.stdout.write(' '.join(map(str, path)) + '\n')


if __name__ == "__main__":
    main()
