import sys


def format_input():
    """
    Formats the input to create a matrix for each row with dimensions MxN where
    M is the first number on each row and N the second number, followed by its elements.
    """
    input_lines = []
    while True:
        line = sys.stdin.readline().strip()
        if line == '':
            break  # Stop if an empty line (newline only) is encountered
        input_lines.append(line)

    matrices = []
    for line in input_lines:
        parts = line.split()
        rows, cols = int(parts[0]), int(parts[1])
        elements = parts[2:]

        if len(elements) != rows * cols:
            print("Invalid matrix dimensions or number of elements. Skipping.")
            continue

        # Convert each element to a float
        matrix = [[float(elements[i * cols + j]) for j in range(cols)] for i in range(rows)]
        matrices.append(matrix)

    return matrices


def format_output(matrix):
    """
    Formats the output to include the dimensions of the matrix followed by its elements.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    formatted_output = f"{rows} {cols} " + " ".join(f"{num:.6f}" for num in matrix[0])
    return formatted_output


def matrix_multiply(a, b):
    result = [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    return result


def main():
    # Read input and create matrices
    matrices = format_input()

    # Extract the transition, emission, and initial state matrices
    transition, emission, initial_state = matrices

    # Perform pi*A multiplication
    intermediate_matrix = matrix_multiply(initial_state, transition)

    # Perform Intermediate*B multiplication
    final_matrix = matrix_multiply(intermediate_matrix, emission)

    # Format the output to match the specifications
    formatted_output = format_output(final_matrix)

    # Print the result to stdout
    sys.stdout.write(formatted_output + '\n')


if __name__ == "__main__":
    main()
