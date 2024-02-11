import sys


def forward_algorithm(A, B, initial_distribution, emissions):
    """
    Uses forward pass to determine probability of seeing a sequence of
    observations O(1:t)
    """
    # Step 1: Initialization
    alpha = [initial_distribution[i] * B[i][emissions[0]] for i in range(len(A))]

    # Step 2: Recursion
    for observation in emissions[1:]:
        alpha_temp = [0] * len(A)
        for j in range(len(A)):
            for i in range(len(A)):
                alpha_temp[j] += alpha[i] * A[i][j]
            alpha_temp[j] *= B[j][observation]
        alpha = alpha_temp

    # Step 3: Termination
    return sum(alpha)


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


def format_output(matrix):
    """
    Formats the output to include the dimensions of the matrix followed by its elements.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    formatted_output = f"{rows} {cols} " + " ".join(f"{num:.6f}" for num in matrix[0])
    return formatted_output


def main():
    # Read and format the input
    matrices, emissions = format_input()

    # Extract the individual matrices
    transition, emission, initial_state = matrices
    initial_distribution = initial_state[0]

    # Calculate the probability of the emission sequence
    probability = forward_algorithm(transition, emission, initial_distribution, emissions)

    # Output the probability
    sys.stdout.write(f"{probability:.6f}\n")


if __name__ == "__main__":
    main()
