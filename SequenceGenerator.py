import numpy as np

class Sequencer(object):
    def __init__(self, SequenceMatrixRowsLength, SequenceMatrixColumnsLength):
        self.SequenceMatrixRowsLength = SequenceMatrixRowsLength
        self.SequenceMatrixColumnsLength = SequenceMatrixColumnsLength

    def Get_Random_Sequence(self, state_list):
        reward = None
        position = 3
        sequence = state_list[position]
        while(True):
            if position == 0 or position == 6:
                reward = 0 if position == 0 else 1
                break

            walk_action = np.random.randint(0, 2)
            position = position - 1 if walk_action == 0 else position + 1

            sequence += state_list[position]
        return sequence, reward 

    def Get_Sequence_Row(self):
        state_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        sequence_list = []
        reward_list = [] 

        iteration = 0
        while(True):
            if iteration == self.SequenceMatrixRowsLength:
                break

            sequence, reward = self.Get_Random_Sequence(state_list)
            sequence_list.append(sequence)

            reward_list.append(reward)

            iteration += 1

        sequence_list = np.array(sequence_list)
        reward_list = np.array(reward_list)
        return sequence_list, reward_list

    def Get_Sequence_Matrix(self):
        sequence_matrix = []
        reward_matrix = []
        
        iteration = 0
        while(True):
            if iteration == self.SequenceMatrixColumnsLength:
                break
            sequence_list, reward_list = self.Get_Sequence_Row()
            sequence_matrix.append(sequence_list)
            reward_matrix.append(reward_list)
            iteration += 1
        
        sequence_matrix = np.array(sequence_matrix)
        reward_matrix = np.array(reward_matrix)
        return sequence_matrix, reward_matrix

    def Vectorize_Sequence(self, sequence):
        vectors = {
            'B': np.array([[1, 0, 0, 0, 0]]).T,
            'C': np.array([[0, 1, 0, 0, 0]]).T,
            'D': np.array([[0, 0, 1, 0, 0]]).T,
            'E': np.array([[0, 0, 0, 1, 0]]).T,
            'F': np.array([[0, 0, 0, 0, 1]]).T
        }
        vectorized_sequence = []
        
        clipped_sequence = sequence[:-1]
        for char in clipped_sequence:
            vectorized_sequence.append(vectors[char])

        vectorized_sequence = np.array(vectorized_sequence)
        return vectorized_sequence

    def Vectorize_Sequence_Matrix(self, sequence_matrix):
        vector_matrix = []

        for row in sequence_matrix:

            vectorized_row = []
            for sequence in row:
                vectorized_row.append(self.Vectorize_Sequence(sequence))
            
            vectorized_row = np.array(vectorized_row)
            vector_matrix.append(vectorized_row)
        
        vector_matrix = np.array(vector_matrix)
        return vector_matrix