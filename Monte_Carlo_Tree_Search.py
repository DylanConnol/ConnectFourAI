import math
import pickle
import numpy as np
import time
from ConnectFourBoard import Board
import random
from NeuralNetwork import NeuralNetwork

import copy

class MCTS:
    def __init__(self, board, checker, num_iterations, exploration_value):
        self.win_reward = 10
        self.loss_penalty = -10
        self.neutrality = 0
        self.num_iterations = num_iterations
        self.ev = exploration_value
        self.current_iteration = 0
        self.height = board.height
        self.width = board.width
        self.checker = checker
        #THIS VALUE SHOULD NOT CHANGE:
        self.main_board = Board(self.height, self.width)
        self.main_board.slots = copy.deepcopy(board.slots)
        #create a main board object so we don't end up accidentally creating a TON of new classes
        self.first_board = board

    def ucb1(self, v, ni):
        if ni == 0: return float('inf')
        return v/ni + self.ev*math.sqrt(math.log(self.current_iteration) / ni)


    def get_child_nodes(self, string, checker):
        self.main_board.from_string(string)
        # checker = "O"*(checker == "X") + "X"*(checker == "O")
        child_nodes = []
        for i in range(self.width):
            if self.main_board.can_add_to(i):
                self.main_board.add_checker(checker, i)
                child_nodes.append(self.main_board.to_string())
                self.main_board.remove_checker(i)
        return child_nodes


    def convert_to_NN_readable(self, board):
        board = [[0] if x == ' ' else [1] if x == 'X' else [-1] for x in board]
        x = np.array(board*3)
        return x


    def evaluate_node_while_using_NN(self, checker, nodes, node, NN):
        inverse_checker = 'O' if checker == 'X' else 'X'
        if nodes[node] == [0, 0]:
            # nodes[node] = [0,0]
            answer = NN.forwardprop((self.convert_to_NN_readable(node)))
            nodes[node] = [answer, 1]
            return answer
        children = self.get_child_nodes(node, checker)
        for child in children:
            if child not in nodes:
                nodes[child] = [0, 0]
                answer = self.evaluate_node_while_using_NN(inverse_checker, nodes, child, NN)
                nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
                return answer
        next_node = max(children, key=lambda x: self.ucb1(nodes[x][0], nodes[x][1]))
        answer = self.evaluate_node_while_using_NN(inverse_checker, nodes, next_node, NN)
        nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
        return answer



    def evaluate_node_while_updating_NN(self, checker, nodes, node, NN, learning_rate):
        inverse_checker = 'O' if checker == 'X' else 'X'
        if nodes[node] == [0,0]:
            # nodes[node] = [0,0]
            self.main_board.from_string(node)
            randomval = self.main_board.run_game_randomly(checker)
            answer = self.win_reward if randomval == checker else self.neutrality if randomval == "Tie" else self.loss_penalty
            nodes[node] = [answer, 1]
            loss = NN.gradient_descent([(self.convert_to_NN_readable(node), self.ucb1(nodes[node][0],nodes[node][1]))], learning_rate)
            return (-1*answer, loss)
        children = self.get_child_nodes(node, checker)
        sm = 0; count = 0
        for child in children:
            if child not in nodes:
                nodes[child] = [0,0]
                answer, loss = self.evaluate_node_while_updating_NN(inverse_checker, nodes, child, NN, learning_rate)
                nodes[node] = [(sm + answer) / (1 + count), nodes[node][1] + 1]
                new_loss = NN.gradient_descent(
                    [(self.convert_to_NN_readable(node), self.ucb1(nodes[node][0], nodes[node][1]))],
                    learning_rate)
                return (-1*answer, loss + new_loss)
            sm += nodes[child][0]; count += 1
        next_node = max(children, key= lambda x: self.ucb1(nodes[x][0],nodes[x][1]))
        next_node_val = self.ucb1(nodes[next_node][0],nodes[next_node][1])
        answer, loss = self.evaluate_node_while_updating_NN(inverse_checker, nodes, next_node, NN, learning_rate)
        nodes[node] = [(sm + answer - next_node_val) / (count), nodes[node][1] + 1]
        new_loss = NN.gradient_descent([(self.convert_to_NN_readable(node), self.ucb1(nodes[node][0], nodes[node][1]))],
                            learning_rate)
        return (-1*answer, loss + new_loss)


    def evaluate_node(self, checker, nodes, node):
        inverse_checker = 'O' if checker == 'X' else 'X'
        if nodes[node] == [0,0]:
            # nodes[node] = [0,0]
            self.main_board.from_string(node)
            randomval = self.main_board.run_game_randomly(checker)
            answer = self.win_reward if randomval == checker else self.neutrality if randomval == "Tie" else self.loss_penalty
            nodes[node] = [answer, 1]
            return -1*answer
        children = self.get_child_nodes(node, checker)
        for child in children:
            if child not in nodes:
                nodes[child] = [0,0]
                answer = self.evaluate_node(inverse_checker, nodes, child)
                nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
                return -1*answer
        next_node = max(children, key= lambda x: self.ucb1(nodes[x][0],nodes[x][1]))
        answer = self.evaluate_node(inverse_checker, nodes, next_node)
        nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
        return -1*answer


    def run_tree_search(self):
        # child_directory = {}
        nodes = {}
        board = self.first_board.to_string()
        # child_directory[board] = self.get_child_nodes(board, self.checker)
        nodes[board] = [0,0]
        for i in range(self.num_iterations):
            self.current_iteration += 1
            self.evaluate_node(self.checker, nodes, board)
        answer =  max(self.get_child_nodes(board,self.checker), key= lambda x: -1*self.ucb1(nodes[x][0],nodes[x][1]))
        self.first_board.from_string(answer)


    def run_tree_search_while_updating_NN(self, seconds_to_run_for, filename, learning_rate, print_and_save_rate = 5000):
        neurons_in_layers = ((self.height * self.width * 3, 100, 25, 1))
        NN = NeuralNetwork(neurons_in_layers)
        nodes = {}
        board = self.first_board.to_string()
        nodes[board] = [0, 0]
        current_time = time.time()
        t = 0; sm = 0
        while True:
            self.current_iteration += 1
            useless, loss = self.evaluate_node_while_updating_NN(self.checker, nodes, board, NN, learning_rate)
            new_time = time.time()
            if new_time - current_time >= seconds_to_run_for: break
            t += 1
            if t % print_and_save_rate == 0:
                t = 0;
                print("Loss: ", sm/100);
                sm = 0
                with open(filename, 'wb') as saves:
                    pickle.dump(NN, saves)
            sm += loss
        # answer = max(self.get_child_nodes(board, self.checker), key=lambda x: -1 * self.ucb1(nodes[x][0], nodes[x][1]))
        # self.first_board.from_string(answer)
        with open(filename, 'wb') as saves:
            pickle.dump(NN, saves)

    def run_tree_search_with_neural_network(self, filename):
        # child_directory = {}
        with open(filename, 'rb') as loads:
            NN = pickle.load(loads)
        nodes = {}
        board = self.first_board.to_string()
        # child_directory[board] = self.get_child_nodes(board, self.checker)
        nodes[board] = [0,0]
        for i in range(self.num_iterations):
            self.current_iteration += 1
            self.evaluate_node_while_using_NN(self.checker, nodes, board, NN)
        answer = min(self.get_child_nodes(board, self.checker), key = lambda x: nodes[x])
        self.first_board.from_string(answer)


if __name__ == "__main__":
    b1 = Board(6, 7)
    mct = MCTS(b1, "X", 1000, 25)
    mct.run_tree_search_while_updating_NN(1000, "NeuralNetwork.pickle", 0.01)
    print("Finished!")
    while True:
        rival_mcts = MCTS(b1, "O", 10000, 25)
        mct.run_tree_search_with_neural_network("NeuralNetwork.pickle")
        print(b1)
        if b1.is_win_for("X"):
            print("My bot has won! Horray!"); break
        rival_mcts.run_tree_search()
        if b1.is_win_for("O"):
            print("My bot has lost :(")
            print(b1)
            break
