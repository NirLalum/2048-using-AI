import logic
import random
from AbstractPlayers import *
import time
from typing import Tuple
import numpy as np
import copy
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move: # iterate over all the moving options (Move.UP, MOVE.Down, etc...)
            new_board, done, score = commands[move](board) # make a move
            if done:
                optional_moves_score[move] = score # save all optional scores

        return max(optional_moves_score, key=optional_moves_score.get) # return the max according to the values (scores)
                                                                       # the key is an object function. in this line its the function get 
                                                                       # which return the values in the dictionary

class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """
    def get_indices(self, board, value, time_limit) -> Tuple[(int, int)]:
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = 1.5*score + self.h1(new_board, 1/1000) + self.h2(new_board, 1) + self.h3(new_board,1) # the 1/1000 is for scaling
                #print(f"score: {score}")
        return max(optional_moves_score, key=optional_moves_score.get)



    # evaluate the ordering of the board: the target is that the higher the tile's score is the upper and lefter 
    # the tile's position in the board is
    def h1(self, board, w1):
        board_copy = np.array(board.copy())
        weight_matrix = np.array([[7, 6, 5, 4], [6, 5, 4, 3], [5, 4, 3, 2],[4, 3, 2, 1]])
        score_matrix = weight_matrix*board_copy
        #print(f"h1: {w1*np.sum(score_matrix)}")
        return w1*np.sum(score_matrix)

    # get a reward for empty cells in the board:    
    def h2(self, board, w2):
        board_copy = np.array(board.copy())
        non_zero_count = np.count_nonzero(board_copy)
        #print(f"h2: {w2*(board_copy.shape[0] * board_copy.shape[1] - non_zero_count)}")
        return w2*(board_copy.shape[0] * board_copy.shape[1] - non_zero_count)

    # huristic: penalty for not decending rows and cols
    def h3(self, board, w3):
        penalty = 0
        size = len(board) - 1
        for i, row in enumerate(board):
            for j, value in enumerate(row):
                if j!=size and board[i][j] < board[i][j+1]:
                    penalty -= 1
                if i!=size and board[i][j] < board[i+1][j]:
                    penalty -= 1
        #print(f"h3: {w3*penalty}")
        return w3*penalty

                
                   
# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        

    def get_move(self, board, time_limit) -> Move:
        #start = time.time()
        _,_,best_move = self.maximize(board, depth = 4)
        #end = time.time()
        #print(end - start)
        return best_move

    # check if reached to a terminal state
    def isTerminal(self, board):
        flag = True
        
        for move in Move:
            _, done, _ = commands[move](board) # new board is the new child
            if done:
                flag = False

        return flag

    # define the max node
    def maximize(self, board, depth):
        (max_child, max_utility, max_move) = (None, -1000000, None)
        utility = 0
        
        if depth == 0 or self.isTerminal(board):
            utility = self.h3(board, 6) + self.h4(board, 1)
            return (None, utility, None)
        
        depth -= 1

        # check al children of the current node
        for move in Move:
            new_board, done, score = commands[move](board) # new board is the new child
            if done:
                (_, utility) = self.minimize(new_board, depth) # minimize will get the utility for each child
                if utility > max_utility:
                    (max_child, max_utility, max_move) = (new_board, utility, move)
        
        return (max_child, max_utility, max_move)

    # define the min node
    def minimize(self, board, depth):
        (min_child, min_utility) = (None, 1000000)
        utility = 0

        if depth == 0 or self.isTerminal(board):
            #utility = self.h1(board, 0) + self.h2(board, 0) + self.h3(board, 6) + self.h4(board, 1)
            utility = self.h3(board, 6) + self.h4(board, 1)
            return (None, utility)

        depth -= 1

        for i,_ in enumerate(board):
            for j,_ in enumerate(board):
                if board[i][j] !=0:
                    continue
                new_board = copy.deepcopy(board)
                new_board[i][j] = 2 # create a child
                (_, utility, _) = self.maximize(new_board, depth)
                if utility < min_utility:
                    (min_child, min_utility) = (new_board, utility)

        return (min_child, min_utility)


    # evaluate the ordering of the board: the target is that the higher the tile's score is the upper and lefter 
    # the tile's position in the board is
    def h1(self, board, w1):
        board_copy = np.array(board.copy())
        weight_matrix = np.array([[7, 6, 5, 4], [6, 5, 4, 3], [5, 4, 3, 2],[4, 3, 2, 1]])
        #weight_matrix = np.array([[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0]])
        score_matrix = weight_matrix*board_copy
        #print(f"h1: {w1*np.sum(score_matrix)}")
        return w1*np.sum(score_matrix)

    # get a reward for empty cells in the board:    
    def h2(self, board, w2):
        board_copy = np.array(board.copy())
        non_zero_count = np.count_nonzero(board_copy)
        #print(f"h2: {w2*(board_copy.shape[0] * board_copy.shape[1] - non_zero_count)}")
        return w2*(board_copy.shape[0] * board_copy.shape[1] - non_zero_count)

    # huristic: penalty for not decending rows and cols
    def h3(self, board, w3):
        start = time.time()
        penalty = 0
        size = len(board) - 1
        for i, row in enumerate(board):
            for j, value in enumerate(row):
                if j!=size and board[i][j] < board[i][j+1]:
                    penalty -= 1 * board[i][j+1]/(board[i][j]+1)
                if i!=size and board[i][j] < board[i+1][j]:
                    penalty -= 1 * board[i+1][j]/(board[i][j]+1)
        #print(f"h3: {w3*penalty}")
        end = time.time()
        print(end - start)
        return w3*penalty


    def h4(self, board, w4):
        board_sum = 0
        zeros_num = 1
        for row in board:
            board_sum += sum(row)
            for value in row:
                if value == 0:
                    zeros_num += 1

        return w4*(board_sum/5*zeros_num)


# part C : alfa beta pruning algo
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.score = 0
        self.start_time = 0
        

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        self.start_time = time.time()
        _,best_move = self.maximize(board, depth = 6, a = float('-inf'), b = float('inf'))
        return best_move

    # TODO: add here helper functions in class, if needed

    def isTerminal(self, board):
        flag = True
        
        for move in Move:
            _, done, _ = commands[move](board) # new board is the new child
            if done:
                flag = False

        return flag

    def maximize(self, board, depth, a, b):
        ( max_utility, max_move) = (float('-inf'), None)
        utility = 0
        
        if depth == 0 or self.isTerminal(board) or time.time() - self.start_time > 0.95:
            #utility = self.h1(board, 0) + self.h2(board, 0) + self.h3(board, 6) + self.h4(board, 1)
            utility = self.h1(board, 1) + self.h5(board, 40)
            utility /= self.h4(board,1)
            #utility = self.h1(board, 1)
            return (utility, None)
        
        depth -= 1
  
        for move in Move:
            new_board, done, self.score = commands[move](board) # new board is the new child
            if done:
                utility = self.minimize(new_board, depth, a, b) # minimize will get the utility for each child
                if utility > max_utility:
                    (max_utility, max_move) = (utility, move)

                if max_utility >= b:
                    return (max_utility, max_move)

                a = max(a, max_utility)

        return (max_utility, max_move)



    def minimize(self, board, depth, a, b):
        min_utility = float('inf')
        utility = 0

        if depth == 0 or self.isTerminal(board) or time.time() - self.start_time > 0.95:
            utility = self.h1(board, 1) + self.h5(board, 40)
            utility /= self.h4(board,1)
            return utility

        depth -= 1

        for i,_ in enumerate(board):
            for j,_ in enumerate(board):
                if board[i][j] !=0:
                    continue
                new_board = copy.deepcopy(board)
                new_board[i][j] = 2 # create a child
                (utility, _) = self.maximize(new_board, depth, a, b)
                if utility < min_utility:
                    min_utility = utility

                if min_utility <= a:
                    return min_utility

                b = min(b, min_utility)

        return min_utility


    # evaluate the ordering of the board: the target is that the higher the tile's score is the upper and lefter 
    # the tile's position in the board is
    def h1(self, board, w1):
        board_np = np.array(board)
        weight_matrix = np.array([[2**12, 2**10, 2**9, 2**8], [2**3, 2**4, 2**5, 2**6], [-2**4, -2**3, -2**2, -2],[-2**4, -2**3, -2**2, -2]])
        score_matrix = weight_matrix*board_np
        return w1*np.sum(score_matrix)

    # get a reward for empty cells in the board:    
    def h2(self, board, w2):
        board_copy = np.array(board.copy())
        non_zero_count = np.count_nonzero(board_copy)
        #print(f"h2: {w2*(board_copy.shape[0] * board_copy.shape[1] - non_zero_count)}")
        return w2*(board_copy.shape[0] * board_copy.shape[1] - non_zero_count)

    # huristic: penalty for not decending rows and cols
    def h3(self, board, w3):
        penalty = 0
        size = len(board) - 1
        for i, row in enumerate(board):
            for j, value in enumerate(row):
                if board[i][j] == 0:
                    continue
                if j!=size and board[i][j] < board[i][j+1]:
                    penalty -= 1 * board[i][j+1]/(board[i][j])
                if i!=size and board[i][j] < board[i+1][j]:
                    penalty -= 1 * board[i+1][j]/(board[i][j])
        #print(f"h3: {w3*penalty}")
        
        return w3*penalty

        # huristic: snake huristic!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def h5(self, board, w3):
        penalty = 0
        size = len(board) - 1
        for i, row in enumerate(board):
            for j, value in enumerate(row):
                
                if i%2 == 0:
                    if j!=size and board[i][j] < board[i][j+1]:
                        if board[i][j] == 0  or board[i][j+1] != 0:
                            continue
                        penalty -= 1 * board[i][j+1]/(board[i][j] + 1)
                else:
                    if j!=size and board[i][j] > board[i][j+1]:
                        if board[i][j] == 0  or board[i][j+1] != 0:
                            continue
                        penalty -= 1 * board[i][j]/(board[i][j+1]+1)

        return w3*penalty

    
    def h4(self, board, w4):
        board_sum = 0
        zeros_num = 1
        for i,row in enumerate(board):
            if i == 0 or i == 1:
                continue
            board_sum += sum(row)
            for value in row:
                if value != 0:
                    zeros_num += 1
        return w4 *  zeros_num
        #return w4*(board_sum/5*zeros_num)


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.p2 = 0.9
        self.p4 = 0.1
        self.start_time = 0

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        self.start_time = time.time()
        _,best_move = self.maximize(board, depth = 4) # 5 was best
        return best_move

    def isTerminal(self, board):
        flag = True
        
        for move in Move:
            _, done, _ = commands[move](board) # new board is the new child
            if done:
                flag = False

        return flag

    def maximize(self, board, depth):
        (max_utility, max_move) = (float('-inf'), None)
        utility = 0
        
        if depth == 0 or self.isTerminal(board) or time.time() - self.start_time > 0.95:
            utility = self.h1(board, 1) + self.h5(board, 40)
            utility /= self.h4(board,1)  
            return (utility, None)
        
        depth -= 1
  
        for move in Move:
            new_board, done, _ = commands[move](board) # new board is the new child
            if done:
                utility = self.expect(new_board, depth) # minimize will get the utility for each child
                if utility > max_utility:
                    (max_utility, max_move) = (utility, move)
        
        if max_move == None:
            print('what')
        return (max_utility, max_move)



    def expect(self, board, depth):
        utility = 0
        avg_utility = 0
        empty_cells_num = self.h1(board, 1)
        p_cell = empty_cells_num/16
        
        if depth == 0 or self.isTerminal(board) or time.time() - self.start_time > 0.95:
            utility = self.h1(board, 1) + self.h5(board, 40)
            utility /= self.h4(board,1) 
            return utility

        depth -= 1

        for i,_ in enumerate(board):
            for j,_ in enumerate(board):
                if board[i][j] !=0:
                    continue
                new_board = copy.deepcopy(board)
                new_board[i][j] = 2 # create a child
                (utility, _) = self.maximize(new_board, depth)
                avg_utility += p_cell*self.p2*utility
                new_board = copy.deepcopy(board)
                new_board[i][j] = 4 # create a child
                (utility, _) = self.maximize(new_board, depth)
                avg_utility += p_cell*self.p4*utility
                

        return avg_utility

     # evaluate the ordering of the board: the target is that the higher the tile's score is the upper and lefter 
    # the tile's position in the board is
    def h1(self, board, w1):
        board_np = np.array(board)
        weight_matrix = np.array([[2**12, 2**10, 2**9, 2**8], [2**3, 2**4, 2**5, 2**6], [-2**4, -2**3, -2**2, -2],[-2**4, -2**3, -2**2, -2]])
        score_matrix = weight_matrix*board_np
        return w1*np.sum(score_matrix)

    # huristic: penalty for not decending rows and cols
    def h3(self, board, w3):
        penalty = 0
        size = len(board) - 1
        for i, row in enumerate(board):
            for j, value in enumerate(row):
                if board[i][j] == 0:
                    continue
                if j!=size and board[i][j] < board[i][j+1]:
                    penalty -= 1 * board[i][j+1]/(board[i][j])
                if i!=size and board[i][j] < board[i+1][j]:
                    penalty -= 1 * board[i+1][j]/(board[i][j])
        
        return w3*penalty

    
    def h4(self, board, w4):
        board_sum = 0
        zeros_num = 1
        for i,row in enumerate(board):
            if i == 0 or i == 1:
                continue
            board_sum += sum(row)
            for value in row:
                if value != 0:
                    zeros_num += 1
        return w4 *  zeros_num


    # huristic: snake huristic 
    def h5(self, board, w3):
        penalty = 0
        size = len(board) - 1
        for i, row in enumerate(board):
            for j, value in enumerate(row):
                
                if i%2 == 0:
                    if j!=size and board[i][j] < board[i][j+1]:
                        if board[i][j] == 0  or board[i][j+1] != 0:
                            continue
                        penalty -= 1 * board[i][j+1]/(board[i][j] + 1)
                else:
                    if j!=size and board[i][j] > board[i][j+1]:
                        if board[i][j] == 0  or board[i][j+1] != 0:
                            continue
                        penalty -= 1 * board[i][j]/(board[i][j+1]+1)

        return w3*penalty


    

