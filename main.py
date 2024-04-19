import numpy as np
import math
import matplotlib.pyplot as plt

def read_input(file_name):
    f = open(file_name, 'r')
    lines = [line.strip() for line in f.readlines()]
    population_size = int(lines[0])
    x_interval, y_interval = [float(x) for x in lines[1].split()]
    a0, a1, a2 = [float(x) for x in lines[2].split()]
    precision = int(lines[3])
    crossover_probability = float(lines[4])
    mutation_probability = float(lines[5])
    N = int(lines[6])
    return population_size, x_interval, y_interval, a0, a1, a2, precision, crossover_probability, mutation_probability, N


def generate_population(population_dimension, x_interval, y_interval, precision):
    X = []
    format_string = '{:.' + str(precision) + 'f}'
    for _ in range(population_dimension):
        x = float(format_string.format(np.random.uniform(x_interval, y_interval)))
        X.append(x)
    return X


def calculate_number_of_bytes(x_interval, y_interval, precision):
    return int(math.log2((y_interval - x_interval) * (10 ** precision))) + 1


def calculate_discretion_step(x_interval, y_interval, length):
    return (y_interval - x_interval) / (2 ** length)


def code(x, x_interval, discretion_step, length):
    index = int((x - x_interval) / discretion_step)
    return bin(index).lstrip('0b').zfill(length)


def decode(b, x_interval, discretion_step, precision):
    precision = '.' + str(precision) + 'f'
    index = int(b, 2)
    return float(format(x_interval + index * discretion_step, precision))


def calculate_fitness(x, a0, a1, a2):
    return a0 + a1 * x + a2 * (x ** 2)


def find_elite(X, B, F, population_size):
    max_fitness = float('-inf')
    elite_x, elite_b, elite_f = 0, 0, 0
    for i in range(population_size):
        if F[i] > max_fitness:
            max_fitness = F[i]
            elite_x = X[i]
            elite_b = B[i]
            elite_f = F[i]
    return elite_x, elite_b, elite_f


def calculate_selection_probability(f, F_sum):
    return f / F_sum


def calculate_selection_intervals(SP):
    return np.concatenate(([0], np.cumsum(SP)))


def generate_selection(SI, population_size):
    U = np.random.rand(population_size)
    selection = []
    for u in U:
        index = binary_search(u, SI)
        selection.append(index)
    return U, selection


def apply_selection(selection, X, B, F):
    new_X, new_B, new_F = [], [], []
    for index in selection:
        new_X.append(X[index])
        new_B.append(B[index])
        new_F.append(F[index])
    return new_X, new_B, new_F


def generate_crossover_participants(crossover_probability, population_size):
    U = np.random.rand(population_size)
    crossover_participants = []
    for i in range(population_size):
        if U[i] < crossover_probability:
            crossover_participants.append(i)
    return U, crossover_participants


def crossover2(b0, b1, index):
    return b0[:index] + b1[index:], b1[:index] + b0[index:]


def crossover3(b0, b1, b2, index):
    return b0[:index] + b1[index:], b1[:index] + b2[index:], b2[:index] + b0[index:]


def apply_crossover(crossover_participants, X, B, F, length, x_interval, discretion_step, precision, a0, a1, a2,
                    output_file=''):
    n = len(crossover_participants)
    # there must be at least two chromosomes in order to have cross over
    if n < 2:
        return
    for i in range(0, n, 2):
        if i != n - 3:
            # the two chromosomes that will be crossed over
            i0 = crossover_participants[i]
            i1 = crossover_participants[i + 1]
            index = int(np.random.uniform(0, length + 1))
            c0, c1 = crossover2(B[i0], B[i1], index)
            if output_file != '':
                print_crossover2(output_file, i0, i1, B[i0], B[i1], index, c0, c1)
            # the two children will replace their parents in the population
            B[i0] = c0
            B[i1] = c1
            X[i0] = decode(c0, x_interval, discretion_step, precision)
            X[i1] = decode(c1, x_interval, discretion_step, precision)
            F[i0] = calculate_fitness(X[i0], a0, a1, a2)
            F[i1] = calculate_fitness(X[i1], a0, a1, a2)
        else:
            # the last three chromosomes will be crossed over
            i0 = crossover_participants[i]
            i1 = crossover_participants[i + 1]
            i2 = crossover_participants[i + 2]
            index = int(np.random.uniform(0, length + 1))
            c0, c1, c2 = crossover3(B[i0], B[i1], B[i2], index)
            # the three children will replace their parents in the population
            if output_file != '':
                print_crossover3(output_file, i0, i1, i2, B[i0], B[i1], B[i2], index, c0, c1, c2)
            B[i0] = c0
            B[i1] = c1
            B[i2] = c2
            X[i0] = decode(c0, x_interval, discretion_step, precision)
            X[i1] = decode(c1, x_interval, discretion_step, precision)
            X[i2] = decode(c2, x_interval, discretion_step, precision)
            F[i0] = calculate_fitness(X[i0], a0, a1, a2)
            F[i1] = calculate_fitness(X[i1], a0, a1, a2)
            F[i2] = calculate_fitness(X[i2], a0, a1, a2)
            break


def apply_mutation(X, B, F, mutation_probability, population_size, length, x_interval, discretion_step, precision, a0,
                   a1, a2):
    mutated = []
    U = np.random.rand(population_size, length)
    for i in range(population_size):
        mutation = False
        for j in range(length):
            if U[i][j] < mutation_probability:
                # flip the bit and update
                mutation = True
                b_copy = [int(b) for b in B[i]]
                b_copy[j] = (b_copy[j] + 1) % 2
                b_copy = "".join(str(b) for b in b_copy)
                B[i] = b_copy
        if mutation:
            X[i] = decode(B[i], x_interval, discretion_step, precision)
            F[i] = calculate_fitness(X[i], a0, a1, a2)
            mutated.append(i)
    return mutated


def binary_search(u, SI):
    left, right = 0, len(SI) - 1
    while left <= right:
        mid = (left + right) // 2
        if SI[mid] <= u < SI[mid + 1]:
            return mid
        elif SI[mid] > u:
            right = mid - 1
        elif u > SI[mid + 1]:
            left = mid + 1


def print_population(output_file, X, B, F, population_size):
    for i in range(population_size):
        output_file.write('{:>2}: {:<25}  x = {:<10}  f = {:<20}\n'.format(i + 1, B[i], X[i], F[i]))


def print_selection_probabilities(output_file, SP, population_size):
    output_file.write('\nProbabilitati selectie\n')
    for i in range(population_size):
        output_file.write('cromozom {:>2}:  probabilitate = {:.12f}\n'.format(i + 1, SP[i]))


def print_selection_intervals(output_file, SI):
    output_file.write('\nIntervale probabilitati selectie\n')
    for si in SI:
        output_file.write(str(si) + '\n')


def print_selection(output_file, U, selection):
    output_file.write('\n')
    for i in range(len(selection)):
        output_file.write('u = {:<.12f} => selectam cromozomul {}\n'.format(U[i], selection[i] + 1))


def print_crossover_participants(output_file, U, B, crossover_probability):
    output_file.write('\nProbabilitatea de incrucisare = ' + str(crossover_probability) + '\n')
    for i in range(len(U)):
        if U[i] < crossover_probability:
            output_file.write(
                '{:>2}: {}  u = {:<.12f} < {} => participa\n'.format(i + 1, B[i], U[i], crossover_probability))
        else:
            output_file.write('{:>2}: {}  u = {:<.12f}\n'.format(i + 1, B[i], U[i]))


def print_crossover2(output_file, i0, i1, b0, b1, index, c0, c1):
    output_file.write('\nRecombinare dintre cromozomul {} cu cromozomul {}:\n'.format(i0 + 1, i1 + 1))
    output_file.write('{}  {}  punct {}\n'.format(b0, b1, index))
    output_file.write('Rezultat  {}  {}\n'.format(c0, c1))


def print_crossover3(output_file, i0, i1, i2, b0, b1, b2, index, c0, c1, c2):
    output_file.write(
        '\nRecombinare dintre cromozomul {} cu cromozomul {} si cromozomul {}:\n'.format(i0 + 1, i1 + 1, i2 + 1))
    output_file.write('{}  {}  {}  punct {}\n'.format(b0, b1, b2, index))
    output_file.write('Rezultat  {}  {}  {}\n'.format(c0, c1, c2))


def print_mutation(output_file, mutated, mutation_probability):
    output_file.write('\nProbabilitatde de mutatie pentru fiecare gena = {}:\n'.format(mutation_probability))
    output_file.write('Au fost modificati cromozomii:\n')
    for i in mutated:
        output_file.write(str(i + 1) + '\n')


population_size, x_interval, y_interval, a0, a1, a2, precision, crossover_probability, mutation_probability, N = read_input(
        'input.txt')
X = generate_population(population_size, x_interval, y_interval, precision)
max_values, mean_values = [], []

# X = [-0.914592, -0.516787, -0.246207, 1.480791, 0.835307, 1.229633, 0.133068, -0.897179, 0.100578, -0.311975, 1.411980, 0.404924, 1.954865, 0.359503, 1.255452, 1.124764, 1.527482, 1.573845, -0.562311, 1.191435]
length = calculate_number_of_bytes(x_interval, y_interval, precision)
discretion_step = calculate_discretion_step(x_interval, y_interval, length)

B, F = [], []
for x in X:
    B.append(code(x, x_interval, discretion_step, length))
    F.append(calculate_fitness(x, a0, a1, a2))

output_file_name = 'evolution.txt'
output_file = open(output_file_name, 'w')
output_file.write('Populatia initiala\n')
print_population(output_file, X, B, F, population_size)

elite_x, elite_b, elite_f = find_elite(X, B, F, population_size)
output_file.write('\nElement elitist: {}  x = {}  f = {}\n'.format(elite_b, elite_x, elite_f))

F_sum = np.sum(F)
SP = []
for f in F:
    SP.append(calculate_selection_probability(f, F_sum))
print_selection_probabilities(output_file, SP, population_size)

SI = calculate_selection_intervals(SP)
print_selection_intervals(output_file, SI)

U, selection = generate_selection(SI, population_size - 1)
print_selection(output_file, U, selection)

X, B, F = apply_selection(selection, X, B, F)
output_file.write('\nDupa selectie\n')
print_population(output_file, X, B, F, population_size - 1)

U, crossover_participants = generate_crossover_participants(crossover_probability, population_size - 1)
print_crossover_participants(output_file, U, B, crossover_probability)

apply_crossover(crossover_participants, X, B, F, length, x_interval, discretion_step, precision, a0, a1, a2,
                    output_file)

output_file.write('\nDupa recombinare:\n')
print_population(output_file, X, B, F, population_size - 1)

mutated = apply_mutation(X, B, F, mutation_probability, population_size - 1, length, x_interval, discretion_step,
                             precision, a0, a1, a2)
print_mutation(output_file, mutated, mutation_probability)
output_file.write('\nDupa mutatie:\n')
print_population(output_file, X, B, F, population_size - 1)

X.append(elite_x)
B.append(elite_b)
F.append(elite_f)
output_file.write('\nDupa adaugarea elementului elitist:\n')
print_population(output_file, X, B, F, population_size)
output_file.write('\n')

max_values.append(np.max(F))
mean_values.append(np.mean(F))

# the rest of the iterations
output_file.write('Evolutia maximului si a mediei:\n')
for i in range(N - 1):
    elite_x, elite_b, elite_f = find_elite(X, B, F, population_size)

    F_sum = np.sum(F)
    SP = []
    for f in F:
        SP.append(calculate_selection_probability(f, F_sum))

    SI = calculate_selection_intervals(SP)

    U, selection = generate_selection(SI, population_size - 1)

    X, B, F = apply_selection(selection, X, B, F)

    U, crossover_participants = generate_crossover_participants(crossover_probability, population_size - 1)

    apply_crossover(crossover_participants, X, B, F, length, x_interval, discretion_step, precision, a0, a1, a2)

    apply_mutation(X, B, F, mutation_probability, population_size - 1, length, x_interval, discretion_step,
                       precision, a0, a1, a2)

    X.append(elite_x)
    B.append(elite_b)
    F.append(elite_f)

    max_values.append(np.max(F))
    mean_values.append(np.mean(F))

    output_file.write('maxim = {:<20}  medie = {:<20}\n'.format(np.max(F), np.mean(F)))

# plot the max and mean values
plt.plot(max_values, label='Maxim')
plt.plot(mean_values, label='Medie')
plt.xlabel('Pas')
plt.ylabel('Valori maxim si medie')
plt.legend()
plt.show()

output_file.close()
