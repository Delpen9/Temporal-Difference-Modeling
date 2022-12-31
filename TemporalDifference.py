import numpy as np

class TD(object):
    def __init__(self, alpha, lambda_value, initial_weights, initial_error):
        self.alpha = alpha
        self.lambda_value = lambda_value
        self.weights = initial_weights
        self.training_set_weight_list = []
        self.initial_error = initial_error

    class RootMeanSquaredError(object):
        def __init__(self, training_set_weight_list):
            ## From the bottom of Page. 20
            self.weights_expected = np.array([
                [1.0 / 6.0],
                [1.0 / 3.0],
                [1.0 / 2.0],
                [2.0 / 3.0],
                [5.0 / 6.0]
            ])

            self.training_set_weight_list = training_set_weight_list

        def Get_Error(self):
            error = 0
            for weights in self.training_set_weight_list:
                error += np.sqrt(np.mean((weights - self.weights_expected)**2)) / len(self.training_set_weight_list)
            return error

    def Update_Weight_From_Sequence(self, reward, vectorized_sequence, training_set_w):
        e_t = self.initial_error
        sequence_w = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        for k in range(len(vectorized_sequence)):
            x_k = vectorized_sequence[k]

            ## Adapted from the equation at the top of Page. 16 in the Sutton paper
            ## From the top of Page. 14, The gradient of P with respect to W is x_k
            ## ------------------------------------------
            e_t = x_k + self.lambda_value * e_t
            ## ------------------------------------------

            ## Equation 4 on Page. 15
            ## ------------------------------------------
            if k == len(vectorized_sequence) - 1:
                sequence_increment = self.alpha * (reward - np.dot(training_set_w.T, x_k)) * e_t
                sequence_w += sequence_increment

            else:
                x_k_plus_1 = vectorized_sequence[k + 1]
                sequence_increment = self.alpha * (np.dot(training_set_w.T, x_k_plus_1) - np.dot(training_set_w.T, x_k)) * e_t

            ## ------------------------------------------    

        return sequence_w

    def Update_Weight_From_Sequence_Row(self, vectorized_sequence_row, reward_row):
        training_set_w = np.array([[0.5], [0.5], [0.5], [0.5], [0.5]])
        delta_w = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
        initial_training_set_w = training_set_w
        while(True):
            for vectorized_sequence, reward in zip(vectorized_sequence_row, reward_row):
                delta_w += self.Update_Weight_From_Sequence(reward, vectorized_sequence, training_set_w)
            
            training_set_w += delta_w

            ## Stopping tolerance
            weight_improvement = np.abs(np.mean(training_set_w - initial_training_set_w))
            if weight_improvement < 0.001:
                break

        self.training_set_weight_list.append(training_set_w)

        return training_set_w

    def Update_Weight_From_Sequence_Matrix(self, vectorized_sequence_matrix, reward_matrix):
        for vectorized_sequence_row, reward_row in zip(vectorized_sequence_matrix, reward_matrix):
            self.weights += self.Update_Weight_From_Sequence_Row(vectorized_sequence_row, reward_row)

    def Get_Root_Mean_Squared_Error(self):
        rmse = self.RootMeanSquaredError(self.training_set_weight_list)
        error = rmse.Get_Error()
        return error