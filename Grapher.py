import numpy as np
from TemporalDifference import TD
import matplotlib.pyplot as plt

class Grapher(object):
    def __init__(self, lambda_values, alpha_values, initial_weights, initial_error, vector_matrix, reward_matrix):
        self.lambda_values = lambda_values
        self.alpha_values = alpha_values
        self.initial_weights = initial_weights
        self.initial_error = initial_error
        self.vector_matrix = vector_matrix
        self.reward_matrix = reward_matrix
        self.point_list = []

    def Set_Point_Values(self):
        for lambda_value in self.lambda_values:
            for alpha in self.alpha_values:
                temporaldifference = TD(alpha, lambda_value, self.initial_weights, self.initial_error)
                temporaldifference.Update_Weight_From_Sequence_Matrix(self.vector_matrix, self.reward_matrix)
                rmse = temporaldifference.Get_Root_Mean_Squared_Error()
                self.point_list.append([lambda_value, alpha, rmse])
        self.point_list = np.array(self.point_list)

    def Display_Graph(self, x_axis, y_axis, tracks, point_dict, color_list):
        track_values = np.unique(point_dict.get(tracks))
        for track_value, color_value in zip(track_values, color_list[:len(track_values)]):
            track_point_list = self.point_list[np.where(point_dict.get(tracks) == track_value)]
            track_point_list_dict = {'lambda' : track_point_list[:, 0], 'alpha' : track_point_list[:, 1], 'rmse' : track_point_list[:, 2]}
            x_axis_values = track_point_list_dict.get(x_axis)
            y_axis_values = track_point_list_dict.get(y_axis)
            plt.plot(x_axis_values, y_axis_values, '-ok', color = color_value)

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend([tracks + '=' + str(track_value) for track_value in track_values])
        plt.show()

    def Graph(self, x_axis, y_axis, tracks):
        plt.style.use('seaborn-whitegrid')

        point_dict = {'lambda' : self.point_list[:, 0], 'alpha' : self.point_list[:, 1], 'rmse' : self.point_list[:, 2]}
        color_list = ['blue', 'green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'white']

        self.Display_Graph(x_axis, y_axis, tracks, point_dict, color_list)