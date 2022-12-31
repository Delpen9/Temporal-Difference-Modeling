import numpy as np
from SequenceGenerator import Sequencer
from Grapher import Grapher

def main():
    training_set_size = 10
    number_of_training_sets = 100
    sequencer = Sequencer(training_set_size, number_of_training_sets)
    sequence_matrix, reward_matrix = sequencer.Get_Sequence_Matrix()
    vector_matrix = sequencer.Vectorize_Sequence_Matrix(sequence_matrix)

    initial_weights = np.array([[0.5], [0.5], [0.5], [0.5], [0.5]])
    initial_error = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])

    options = ['rmse', 'alpha', 'lambda']

    ## Figure 3
    lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    alpha_values = [0.3]
    grapher_1 = Grapher(lambda_values, alpha_values, initial_weights, initial_error, vector_matrix, reward_matrix)
    grapher_1.Set_Point_Values()
    grapher_1.Graph(options[2], options[0], options[1])

    ## Figure 4
    lambda_values = [0.0, 0.3, 0.8, 1.0]
    alpha_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4]
    grapher_2 = Grapher(lambda_values, alpha_values, initial_weights, initial_error, vector_matrix, reward_matrix)
    grapher_2.Set_Point_Values()
    grapher_2.Graph(options[1], options[0], options[2])

    ## Figure 5
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_values = [0.135]
    grapher_3 = Grapher(lambda_values, alpha_values, initial_weights, initial_error, vector_matrix, reward_matrix)
    grapher_3.Set_Point_Values()
    grapher_3.Graph(options[2], options[0], options[1])

if __name__ == "__main__":
    main()