import math
import pickle
import numpy as np
import time
from ConnectFourClass import Board
import random
from NeuralNetworkTemplate import NeuralNetwork
import csv
import copy

class MCTS:
    def __init__(self, board, checker, num_iterations, exploration_value):
        self.win_reward = 10
        self.loss_penalty = -10
        self.neutrality = 0
        self.loss = 0
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

    def convert_to_NN_readable(self, board, current_checker):
        #one hot key encryption of the board * 2 for cross-referencing or whatever
        new_board = [[0,0,1] if x == ' ' else [0,1,0] if x == 'X' else [1,0,0] for x in board]
        lst = []
        for i in new_board:
            for j in i:
                lst += [j]
        lst *= 2

        #current checker
        lst += [1*(current_checker=="O"), -1*(current_checker=="X")]

        #Convolutional Layers
        #2x2
        new_list = [0]*(self.height-1)*(self.width-1)
        for i in range(self.height):
            for j in range(self.width):
                chip = board[i*self.width + j]
                if i != 0 and j != (self.width-1):
                    new_list[(i-1)*(self.width-1) + j] += 1 if chip == self.checker else -1
                if j != 0 and i != (self.height-1):
                    new_list[(i)*(self.width - 1) + (j-1)] += 1 if chip == self.checker else -1
                if j != 0 and i != 0:
                    new_list[(i-1) * (self.width - 1) + (j - 1)] += 1 if chip == self.checker else -1
        lst += new_list
        #4x4
        new_list = [0] * (self.height - 3) * (self.width - 3)
        for i in range(self.height):
            for j in range(self.width):
                chip = board[i * self.width + j]
                if j + 3 < self.width:
                    for vertical in range(1, min(4, i)):
                        if i - vertical <= self.height - 4:
                            new_list[(i - vertical) * (self.width - 3) + j] += 1 if chip == self.checker else -1
                if i + 3 < self.height:
                    for horizontal in range(1, min(4, j)):
                        if j - horizontal <= self.width - 4:
                            new_list[(i) * (self.width - 3) + (j - horizontal)] += 1 if chip == self.checker else -1
                if i >= 3 and j >= 3:
                    for diagonal in range(-3, 0):
                        if self.width - (j+diagonal) >= 4 and self.height - (i+diagonal) >= 4:
                            new_list[(i + diagonal) * (self.width - 3) + (j + diagonal)] += 1 if chip == self.checker else -1
        lst += new_list

        #playable spaces
        for i in range(self.height):
            for j in range(self.width):
                if (board[i * self.width + j] != ' ' and board[(i - 1) * self.width + j] == ' ') or (
                        i == 0 and board[i * self.width + j] == ' '):
                    lst += [1] if current_checker == self.checker else [-1]
                else:
                    lst += [0]

        x = np.array(lst).reshape((338,1))
        return x


    def evaluate_node_while_using_NN(self, checker, nodes, node, NN):
        # self.main_board.from_string(node)
        # if self.main_board.is_win_for("X"):
        #     if node in nodes:
        #         nodes[node][1] += 1;
        #         return (-1 * nodes[node][0])
        #     elif checker == "X":
        #         nodes[node] = [self.win_reward, 1]; return (-1 * self.win_reward)
        #     else:
        #         nodes[node] = [self.loss_penalty, 1];
        #         return (-1 * self.loss_penalty)
        # elif self.main_board.is_win_for("O"):
        #     if node in nodes:
        #         nodes[node][1] += 1;
        #         return (-1 * nodes[node][0])
        #     elif checker == "O":
        #         nodes[node] = [self.win_reward, 1]; return (-1 * self.win_reward)
        #     else:
        #         nodes[node] = [self.loss_penalty, 1]; return (-1 * self.loss_penalty)
        # elif self.main_board.is_full():
        #     if node in nodes:
        #         nodes[node][1] += 1
        #     else:
        #         nodes[node] = [self.neutrality, 1]
        #     return (-1 * self.neutrality)
        # inverse_checker = 'O' if checker == 'X' else 'X'
        # if nodes[node] == [0, 0]:
        #     # nodes[node] = [0,0]
        #     answer = float(NN.forwardprop((self.convert_to_NN_readable(node, checker)))[0][0])
        #     nodes[node] = [answer, 1]
        #     return -1*answer
        # children = self.get_child_nodes(node, checker)
        # for child in children:
        #     if child not in nodes:
        #         nodes[child] = [0, 0]
        #         answer = self.evaluate_node_while_using_NN(inverse_checker, nodes, child, NN)
        #         nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
        #         return -1*answer
        # next_node = max(children, key=lambda x: self.ucb1(nodes[x][0], nodes[x][1]))
        # answer = self.evaluate_node_while_using_NN(inverse_checker, nodes, next_node, NN)
        # nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
        # return -1*answer
        inverse_checker = 'O' if checker == 'X' else 'X'
        self.main_board.from_string(node)
        if self.main_board.is_win_for("X"):
            if node in nodes:
                nodes[node][1] += 1
                nodes[node][0] += self.win_reward if checker == "X" else self.loss_penalty
                return -1 * (self.win_reward if checker == "X" else self.loss_penalty)
            elif checker == "X":
                nodes[node] = [self.win_reward, 1];
                return -1 * self.win_reward
            else:
                nodes[node] = [self.loss_penalty, 1];
                return -1 * self.loss_penalty
        elif self.main_board.is_win_for("O"):
            if node in nodes:
                nodes[node][1] += 1;
                nodes[node][0] += self.win_reward if checker == "O" else self.loss_penalty
                return -1 * (self.win_reward if checker == "O" else self.loss_penalty)
            elif checker == "O":
                nodes[node] = [self.win_reward, 1];
                return -1 * self.win_reward
            else:
                nodes[node] = [self.loss_penalty, 1];
                return -1 * self.loss_penalty
        elif self.main_board.is_full():
            if node in nodes:
                nodes[node][1] += 1
            else:
                nodes[node] = [self.neutrality, 1]
            return -1 * self.neutrality
        if nodes[node] == [0, 0]:
            # nodes[node] = [0,0]
            answer = float(NN.forwardprop((self.convert_to_NN_readable(node, checker)))[0][0])
            nodes[node] = [answer, 1]
            return -1 * answer
        children = self.get_child_nodes(node, checker)
        for child in children:
            if child not in nodes:
                nodes[child] = [0, 0]
                answer = self.evaluate_node(inverse_checker, nodes, child)
                nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
                return -1 * answer
        next_node = max(children, key=lambda x: self.ucb1(nodes[x][0], nodes[x][1]))
        answer = self.evaluate_node(inverse_checker, nodes, next_node)
        nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
        return -1 * answer



    def evaluate_node_while_updating_NN(self, checker, nodes, node, NN, learning_rate):
        # self.main_board.from_string(node)
        # if self.main_board.is_win_for("X"):
        #     if node in nodes:
        #         nodes[node][1] += 1; return (-1*self.win_reward, 0)
        #     elif checker == "X": nodes[node] = [self.win_reward, 1]; return (-1*self.win_reward, 0)
        #     else:
        #         nodes[node] = [self.loss_penalty, 1]; return (-1*self.loss_penalty, 0)
        # elif self.main_board.is_win_for("O"):
        #     if node in nodes:
        #         nodes[node][1] += 1; return (-1*self.loss_penalty, 0)
        #     elif checker == "O": nodes[node] = [self.win_reward, 1]; return (-1*self.win_reward, 0)
        #     else: nodes[node] = [self.loss_penalty, 1]; return (-1*self.loss_penalty, 0)
        # elif self.main_board.is_full():
        #     if node in nodes:
        #         nodes[node][1] += 1
        #     else:
        #         nodes[node] = [self.neutrality, 1]
        #     return (-1*self.neutrality, 0)
        # inverse_checker = 'O' if checker == 'X' else 'X'
        # if nodes[node] == [0,0]:
        #     # nodes[node] = [0,0]
        #     self.main_board.from_string(node)
        #     randomval = self.main_board.run_game_randomly(checker)
        #     answer = self.win_reward if randomval == checker else self.neutrality if randomval == "Tie" else self.loss_penalty
        #     nodes[node] = [answer, 1]
        #     loss = NN.gradient_descent([(self.convert_to_NN_readable(node, checker), self.ucb1(nodes[node][0],nodes[node][1]))], learning_rate)
        #     return (-1*answer, loss)
        # children = self.get_child_nodes(node, checker)
        # sm = 0; count = 0
        # for child in children:
        #     if child not in nodes:
        #         nodes[child] = [0,0]
        #         answer, loss = self.evaluate_node_while_updating_NN(inverse_checker, nodes, child, NN, learning_rate)
        #         nodes[node] = [(sm + answer) / (1 + count), nodes[node][1] + 1]
        #         new_loss = NN.gradient_descent(
        #             [(self.convert_to_NN_readable(node, checker), self.ucb1(nodes[node][0], nodes[node][1]))],
        #             learning_rate)
        #         return (-1*answer, loss + new_loss)
        #     sm += nodes[child][0]; count += 1
        # next_node = max(children, key= lambda x: self.ucb1(nodes[x][0],nodes[x][1]))
        # next_node_val = self.ucb1(nodes[next_node][0],nodes[next_node][1])
        # answer, loss = self.evaluate_node_while_updating_NN(inverse_checker, nodes, next_node, NN, learning_rate)
        # nodes[node] = [(sm + answer - next_node_val) / (count), nodes[node][1] + 1]
        # new_loss = NN.gradient_descent([(self.convert_to_NN_readable(node, checker), self.ucb1(nodes[node][0], nodes[node][1]))],
        #                     learning_rate)
        # return (-1*answer, loss + new_loss)
        inverse_checker = 'O' if checker == 'X' else 'X'
        self.main_board.from_string(node)
        if self.main_board.is_win_for("X"):
            if node in nodes:
                nodes[node][1] += 1
                nodes[node][0] += self.win_reward if checker == "X" else self.loss_penalty
                return -1 * (self.win_reward if checker == "X" else self.loss_penalty)
            elif checker == "X":
                nodes[node] = [self.win_reward, 1];
                return -1 * self.win_reward
            else:
                nodes[node] = [self.loss_penalty, 1];
                return -1 * self.loss_penalty
        elif self.main_board.is_win_for("O"):
            if node in nodes:
                nodes[node][1] += 1;
                nodes[node][0] += self.win_reward if checker == "O" else self.loss_penalty
                return -1 * (self.win_reward if checker == "O" else self.loss_penalty)
            elif checker == "O":
                nodes[node] = [self.win_reward, 1];
                return -1 * self.win_reward
            else:
                nodes[node] = [self.loss_penalty, 1];
                return -1 * self.loss_penalty
        elif self.main_board.is_full():
            if node in nodes:
                nodes[node][1] += 1
            else:
                nodes[node] = [self.neutrality, 1]
            return -1 * self.neutrality
        if nodes[node] == [0, 0]:
            # nodes[node] = [0,0]
            self.main_board.from_string(node)
            randomval = self.main_board.run_game_randomly(checker)
            answer = self.win_reward if randomval == checker else self.neutrality if randomval == "Tie" else self.loss_penalty
            self.loss = NN.gradient_descent([(self.convert_to_NN_readable(node, checker), answer)], learning_rate)
            nodes[node] = [answer, 1]
            return -1 * answer
        children = self.get_child_nodes(node, checker)
        for child in children:
            if child not in nodes:
                nodes[child] = [0, 0]
                answer = self.evaluate_node_while_updating_NN(inverse_checker, nodes, child, NN, learning_rate)
                nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
                return -1 * answer
        next_node = max(children, key=lambda x: self.ucb1(nodes[x][0], nodes[x][1]))
        answer = self.evaluate_node_while_updating_NN(inverse_checker, nodes, next_node, NN, learning_rate)
        nodes[node] = [nodes[node][0] + answer, nodes[node][1] + 1]
        return -1 * answer

    def evaluate_node(self, checker, nodes, node):
        inverse_checker = 'O' if checker == 'X' else 'X'
        self.main_board.from_string(node)
                if self.main_board.is_win_for("X" if checker == "O" else "O"):
            if node in nodes:
                nodes[node][1] += 1
                nodes[node][0] += self.win_reward if checker == "X" else self.loss_penalty
                return -1 * (self.win_reward if checker == "X" else self.loss_penalty)
            nodes[node] = [self.loss_penalty, 1];
            return -1 * self.loss_penalty
        elif self.main_board.is_full():
            if node in nodes:
                nodes[node][1] += 1
            else:
                nodes[node] = [self.neutrality, 1]
            return -1 * self.neutrality
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


    def run_tree_search(self, checker = None):
        if checker == None: checker = self.checker
        # child_directory = {}
        nodes = {}
        board = self.first_board.to_string()
        # child_directory[board] = self.get_child_nodes(board, self.checker)
        nodes[board] = [0,0]
        for i in range(self.num_iterations):
            self.current_iteration += 1
            self.evaluate_node(checker, nodes, board)
        answer =  max(self.get_child_nodes(board,checker), key= lambda x: -1*self.ucb1(nodes[x][0],nodes[x][1]))
        self.first_board.from_string(answer)
        return nodes[board], board


    def run_tree_search_while_updating_NN(self, seconds_to_run_for, filename, learning_rate, print_rate = 10, repetitions = 500):
        try:
            with open(filename, 'rb') as neuralnet:
                NN = pickle.load(neuralnet)
                print("Retrieved Neural Net!")
        except:
            neurons_in_layers = ((338, 150, 1))
            NN = NeuralNetwork(neurons_in_layers)
        nodes = {}
        board = self.first_board.to_string()
        self.learning_rate = learning_rate
        nodes[board] = [0, 0]
        current_time = time.time()
        t = 0; sm = 0; smnotsquared = 0
        while True:
            self.current_iteration += 1
            for i in range(repetitions):
                self.evaluate_node(self.checker, nodes, board)
            loss = NN.gradient_descent([(self.convert_to_NN_readable(board, self.checker), nodes[board][0])], learning_rate)
            # loss = self.loss
            # useless, loss = self.evaluate_node_while_updating_NN(self.checker, nodes, board, NN, learning_rate)
            self.main_board.reset()
            depth = random.randint(3, 10)
            self.main_board.random_game_until_threshold("X", depth)
            board = self.main_board.to_string()
            nodes = {}
            nodes[board] = [0, 0]

            new_time = time.time()
            if new_time - current_time >= seconds_to_run_for: break
            t += 1
            if t % print_rate == 0:
                t = 0;
                print(f"Average Loss Squared: {sm/(print_rate)} per iteration\nTime Allotted: {new_time - current_time} seconds\n"
                      f"Average Loss (Not Squared): {(smnotsquared/print_rate)}\n");
                sm = 0; smnotsquared = 0
                with open(filename, 'wb') as saves:
                    pickle.dump(NN, saves)
            sm += loss; smnotsquared += math.sqrt(loss)
        # answer = max(self.get_child_nodes(board, self.checker), key=lambda x: -1 * self.ucb1(nodes[x][0], nodes[x][1]))
        # self.first_board.from_string(answer)
        with open(filename, 'wb') as saves:
            pickle.dump(NN, saves)

    def run_tree_search_with_neural_network(self, filename):
        child_directory = {}
        with open(filename, 'rb') as loads:
            NN = pickle.load(loads)
        nodes = {}
        board = self.first_board.to_string()
        child_directory[board] = self.get_child_nodes(board, self.checker)
        nodes[board] = [0,0]
        for i in range(self.num_iterations):
            self.current_iteration += 1
            self.evaluate_node_while_using_NN(self.checker, nodes, board, NN)
        answer = min(self.get_child_nodes(board, self.checker), key = lambda x: self.ucb1(nodes[x][0],nodes[x][1]))
        self.first_board.from_string(answer)


if __name__ == "__main__":
    def assemble_training_data(num_iters, data_limit, filename):
        nodes = []
        with open(filename, mode='r') as file:
            # reading the CSV file
            csvFile = csv.reader(file)
            for row in csvFile:
                nodes += [row]
        b1 = Board(6, 7)
        mct = MCTS(b1, "X", num_iters, 25)
        count = 1
        start = len(nodes)
        percent = start/data_limit * 100
        for i in range(start, data_limit):
            threshold = random.randint(3, 20)
            if threshold % 2: checker = "O"
            else: checker = "X"
            b1.random_game_until_threshold("X", threshold)
            val, node = mct.run_tree_search(checker)
            nodes.append([node, val[0], checker, threshold])
            b1.reset()
            count += 1
            if i/data_limit >= percent/100:
                print(f"{percent}% . . . {i} completed");
                percent += 0.1;
                count = 0
                with open(filename, 'wb') as output:
                    np.savetxt(output, nodes, delimiter=",", fmt = "% s")
        with open(filename, 'wb') as output:
            np.savetxt(output, nodes, delimiter=",", fmt="% s")
        return len(nodes)

    #assemble training data:
    assemble_training_data(100, 100000, 'training_data_rep_100.csv')


    ## Vs person
    b1 = Board(6, 7)
    mct = MCTS(b1, "X", 100, 25)
    rival_mcts = MCTS(b1, "O", 10000, 25)
    #seconds_to_run_for, filename, learning_rate, print_rate = 10, repetitions = 500
    mct.run_tree_search_while_updating_NN(10000, "NeuralNetworkrep10.pickle", 0.00001, print_rate=1000, repetitions=10)
    print("Finished!")
    while True:
        mct.run_tree_search_with_neural_network("NeuralNetwork.pickle")
        print(b1)
        # inp = int(input("Where would you like to move?\n"))
        # b1.add_checker("X", inp)
        if b1.is_win_for("X"):
            print("My bot has won! Horray!"); break
        # rival_mcts.run_tree_search()
        inp = int(input("Where would you like to move?"))
        b1.add_checker("O", inp)
        if b1.is_win_for("O"):
            print("My bot has lost :(")
            print(b1)
            break


    # b1 = Board(6, 7)
    # #Monte Carlo Search / Neural Network benchmark test against random
    # mctTest = MCTS(b1, "X", 100, 25)
    # results_wld = [0, 0, 0]
    # while True:
    #     mctTest.run_tree_search_with_neural_network("NeuralNetwork.pickle")
    #     # mctTest.run_tree_search()
    #     if b1.is_win_for("X"):
    #         results_wld[0] += 1
    #         b1.reset()
    #         print(f"Win Rate: {results_wld[0] / sum(results_wld)}\nLoss Rate: {results_wld[2] / sum(results_wld)}\n"
    #               f"Num Games: {sum(results_wld)}\n")
    #     options = []
    #     for i in range(6):
    #         if b1.can_add_to(col=i):
    #             options += [i]
    #     choice = random.choice(options)
    #     b1.add_checker("O", choice)
    #     if b1.is_win_for("O"):
    #         results_wld[2] += 1
    #         b1.reset()
    #         print(f"Win Rate: {results_wld[0] / sum(results_wld)}\nLoss Rate: {results_wld[2] / sum(results_wld)}\n"
    #               f"Num Games: {sum(results_wld)}\n")
    #     if b1.is_full():
    #         results_wld[1] += 1
    #         print(f"Win Rate: {results_wld[0] / sum(results_wld)}\nLoss Rate: {results_wld[2] / sum(results_wld)}\n"
    #               f"Num Games: {sum(results_wld)}\n")
    #         b1.reset()


    # # NN vs 1000 MCTS
    # b1 = Board(6, 7)
    # mct = MCTS(b1, "X", 100, 25)
    # rival_mcts = MCTS(b1, "O", 1000, 25)
    # results_wld = [0,0,0]
    # # print("Finished!")
    # while True:
    #     mct.run_tree_search_with_neural_network("NeuralNetwork.pickle")
    #     if b1.is_win_for("X"):
    #         results_wld[0] += 1
    #         b1.reset()
    #         print(f"Win Rate for NN: {results_wld[0] / sum(results_wld)}\nLoss Rate for NN: {results_wld[2] / sum(results_wld)}\n"
    #               f"Num Games: {sum(results_wld)}\n")
    #     rival_mcts.run_tree_search()
    #     if b1.is_win_for("O"):
    #         results_wld[2] += 1
    #         b1.reset()
    #         print(f"Win Rate for NN: {results_wld[0] / sum(results_wld)}\nLoss Rate for NN: {results_wld[2] / sum(results_wld)}\n"
    #               f"Num Games: {sum(results_wld)}\n")
    #     if b1.is_full():
    #         results_wld[1] += 1
    #         print(f"Win Rate for NN: {results_wld[0] / sum(results_wld)}\nLoss Rate for NN: {results_wld[2] / sum(results_wld)}\n"
    #               f"Num Games: {sum(results_wld)}\n")
    #         b1.reset()
