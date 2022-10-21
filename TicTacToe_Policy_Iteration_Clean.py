import numpy as np
import random
import pickle
import copy
import time
from datetime import datetime

BOARD_ROWS = 3
BOARD_COLS = 3
FILE_NUM = 500

class State:
    def __init__(self, teacher, student):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board = self.board.astype(str)
        self.teacher = teacher
        self.student = student
        self.isEndGame = False
        
        #impt data to track
        # self.p1Wins = 0
        # self.p1First = 0
        # self.p2Wins = 0
        # self.p2First = 0
        # self.draws = 0

        #testing
        self.state_index = 0
        self.count = 0

    def get_board(self, data):
        clean = data.strip("[]").replace("'", "").split(" ")
        return np.reshape(clean, (3,3))

    def get_hash(self, board):
        hash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return hash

    def get_symbol(self, board):
        array = board.flatten()
        unique, counts = np.unique(array, return_counts=True)

        elements = dict(zip(unique, counts))
        x_count = 0 if elements.get('X') is None else elements.get('X')
        o_count = 0 if elements.get('O') is None else elements.get('O')

        return 'X' if x_count <= o_count else 'O'
    
    def update_state(self, board, position, symbol):
        board_copy = board.copy()
        board_copy[position[0]][position[1]] = symbol
        return board_copy

    def available_positions(self, board):
        available = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if board[i][j] == '0.0':
                    available.append((i,j))
        
        return available
    
    #for returning the reward for computation
    def get_reward(self, board):
        for i in range(BOARD_ROWS):
            #check row
            if 'X' == board[i][0] == board[i][1] and board[i][1] == board[i][2]:
                return 1
            #check col
            if 'X' == board[0][i] == board[1][i] and board[1][i] == board[2][i]:
                return 1

            #return -1 reward if O wins
            #check row
            if 'O' == board[i][0] == board[i][1] and board[i][1] == board[i][2]:
                return -1
            #check col
            if 'O' == board[0][i] == board[1][i] and board[1][i] == board[2][i]:
                return -1

        #diagonal for X
        if 'X' == board[0][0] == board[1][1] and board[1][1] == board[2][2]:
            return 1
        if 'X' == board[2][0] == board[1][1] and board[1][1] == board[0][2]:
            return 1

        #diagonal for O
        if 'O' == board[0][0] == board[1][1] and board[1][1] == board[2][2]:
            return -1
        if 'O' == board[2][0] == board[1][1] and board[1][1] == board[0][2]:
            return -1
        
        #draw
        if len(self.available_positions(board)) == 0:
            return 0.5
        
        #no reward for non-terminating states
        return 0

    def play_game(self, max_policy_iter=10000, max_value_iter=10000):
        self.teacher.load_player_states()

        print("Starting algorithm...")
        for i in range(max_policy_iter):
            if i%10 == 0:
                print(f"i: {i}")
            
            #Policy Evaluation
            print('Policy Evaluation')
            for j in range(max_value_iter):
                if j%100 == 0:
                    print(f"j: {j}")

                max_diff = 0

                countt = 0
                for state in self.teacher.state_values.keys():
                    # print('state', state)
                    board = self.get_board(state)
                    reward = self.get_reward(board) 

                    v_s = 0
                    probability = 1 #just for the sake of formality
                    for move in self.available_positions(board):
                        symbol = self.get_symbol(board)
                        #update state to get next_state
                        next_state = self.update_state(board, move, symbol)
                        hash = self.get_hash(next_state)

                        #check validity of state
                        if not self.is_valid_state(next_state):
                            continue

                        v_s += probability * self.teacher.state_values[hash]
                        # print(f'prob: 1\trew: {reward}\tgamma:{self.teacher.gamma}*{self.teacher.state_values[hash]}')
                        # print(f'probability * (reward + {(self.teacher.gamma * self.teacher.state_values[hash])})')
                        # v_s += probability * (reward + (self.teacher.gamma * self.teacher.state_values[hash]))
                    
                    v_s = reward + (self.teacher.gamma*v_s)
                    
                    countt += 1
                    if countt%1000 == 0:
                        print('state count: ', countt)

                    #get max of the current max diff and the difference between new v_s and the old v_s
                    # print(f'v_s: {v_s} \nv_old: {self.teacher.state_values[state]}\ndiff: {abs(v_s - self.teacher.state_values[state])}')
                    max_diff = max(max_diff, abs(v_s - self.teacher.state_values[state]))
                    self.teacher.state_values[state] = v_s
                
                # print('max_diff', max_diff)
                if max_diff < self.teacher.delta:
                    print('STOP INDEX: ', j)
                    break

            
            #Policy Improvement
            print('Policy Improvement')
            optimal_policy_found = True
            count = 0
            for state in self.teacher.state_values.keys():
                board = self.get_board(state)
                # print('board\n', board)

                #skip states that are invalid
                '''
                    1. Winning state - cant really evaluate
                    2. Full state
                '''
                available_positions = self.available_positions(board)
                is_end_state = self.is_end_state(board, 'X') or self.is_end_state(board,'O')
                if available_positions == [] or is_end_state:
                    continue


                a_list = {}
                action = self.teacher.policy[state]

                reward = self.get_reward(board)
                for a in available_positions:
                    # print('action', a)
                    symbol = self.get_symbol(board)
                    
                    #get new_state when taking action a
                    new_state = self.update_state(board, a, symbol)
                    # print('new state', new_state)

                    v_s = 0
                    probability = 1 #just for the sake of formality

                    new_available_positions = self.available_positions(new_state)
                    new_symbol = 'X' if symbol == 'O' else 'O'
                    for move in new_available_positions:
                        #update state to get next_state
                        next_state = self.update_state(new_state, move, new_symbol)
                        # print('next_state', next_state)
                        next_hash = self.get_hash(next_state)

                        #check validity of state
                        if not self.is_valid_state(next_state):
                            continue

                        # print('s_v of next hash', self.teacher.state_values[next_hash])
                        v_s += probability * self.teacher.state_values[next_hash]
                    
                    v_s = reward + (self.teacher.gamma*v_s)
                    a_list[a] = v_s

                max_num = max(a_list.values())

                #if there are multiple same elements, randomly get 1
                max_list = [index for index in a_list.keys() if a_list[index] == max_num]
                max_action = random.choice(max_list)
                
                #if there's at least 1 action that is not the same, it means the policy is still improving, therefore optimal policy is not found yet
                if action != max_action:
                    optimal_policy_found = False
                    self.teacher.policy[state] = max_action
                # print('a_list', a_list)
                # print('max_action', max_action)
                if count%100==0:
                    print(f"count: {count}")
                count+=1

            # If actions / policy did not change, algorithm terminates
            if optimal_policy_found:
                print("Optimal policy has been found!")
                break
        
            print(f"index i:{i}")
        #save both states and policy
        self.teacher.save_states(f'teacher_trained_states_{FILE_NUM}.txt')
        self.teacher.save_policy(f'teacher_trained_policy_{FILE_NUM}.txt')
    
    def display_board(self):
        for i in range(BOARD_ROWS):
            print("-------------------")
            for j in range(BOARD_COLS):
                if j == 0:
                    print("|  ", end="")
                value = " " if self.board[i][j] == '0.0' else self.board[i][j]
                print(f"{value}  |  ", end="")
            print()
        print("-------------------")


    ### STATE VALIDATION FUNCTIONS
    def is_end_state(self, board, symbol):
        for i in range(BOARD_ROWS):
            #check row
            if symbol == board[i][0] == board[i][1] and board[i][1] == board[i][2]:
                return 1
            
            #check col
            if symbol == board[0][i] == board[1][i] and board[1][i] == board[2][i]:
                return 1

        #diagonal
        if symbol == board[0][0] == board[1][1] and board[1][1] == board[2][2]:
            return 1
        
        if symbol == board[2][0] == board[1][1] and board[1][1] == board[0][2]:
            return 1
        
        #draw
        if len(self.available_positions(board)) == 0:
            return 0.5

        return 0

    def add_new_state_hash(self, hash_key):
        self.teacher.state_values[hash_key] = 0
        self.state_index +=1
        print(self.state_index)
    
    def is_valid_state(self, state, init=False):
        #check if double win
        x_win = self.is_end_state(state, 'X') == 1
        o_win = self.is_end_state(state, 'O') == 1
        if x_win and o_win:
            return False

        #check if hash exists
        #during initialization of states for TTT, we do not include hash keys that already exist in the dict
        #during policy evaluation, valid states are hash keys that exist
        hash_key = self.get_hash(state)
        if not init:
            if hash_key not in self.teacher.state_values:
                return False
        else:
            if hash_key in self.teacher.state_values:
                return False

        return True

    def get_succeeding_states(self, initial_state, symbol):
        '''
        Checklist of invalid states:
        1. non-alternating move
        2. invalid numbre of Xs or Os
        3. two winning states -- never reached
        4. 
        
        '''

        available_positions = self.available_positions(initial_state)

        #base case: if no more possible positions
        if len(available_positions) == 0:
            self.add_new_state_hash(self.get_hash(initial_state))
        else:
            for move in available_positions:
                new_state = initial_state.copy()
                new_state[move] = symbol
                
                #check validity of state
                if not self.is_valid_state(new_state, init=True):
                    continue

                #add valid state
                self.add_new_state_hash(self.get_hash(new_state))
                #traverse through all possibilities of succeeding states
                self.get_succeeding_states(new_state, 'X' if symbol == 'O' else 'O')

    def initialize_player_states(self):
        #adding empty board
        empty_hash = self.get_hash(self.board)
        self.add_new_state_hash(empty_hash)

        #traverse through the board to simulate all possible moves of X at the start
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                x_state = self.board.copy()
                x_state[i][j] = 'X'

                x_hash = self.get_hash(x_state)
                self.add_new_state_hash(x_hash)

                #traverse through all succeeding states, with O as the next symbol
                self.get_succeeding_states(x_state, 'O')

class Teacher:
    def __init__(self, gamma=0.9, delta=0.001):
        self.delta = delta
        self.gamma = gamma
        self.state_values = {}
        self.policy = {} #copy state values and should populate with random actions!!

    def get_board(self, data):
        clean = data.strip("[]").replace("'", "").split(" ")
        return np.reshape(clean, (3,3))

    def available_positions(self, board):
        available = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if board[i][j] == '0.0':
                    available.append((i,j))
        
        return available

    #policy starts randomly
    def get_random_action(self, hash_key):
        #clean board, get available positions
        board = self.get_board(hash_key)
        available_positions = self.available_positions(board)

        #if it's already a winning state, no action needed
        x_win = self.is_end_state(board, 'X')
        o_win = self.is_end_state(board, 'O')
        win = x_win or o_win

        if len(available_positions) > 0 and not win:
            action = random.choice(available_positions) ##action
        else:
            action = None
        return action
    
    def update_policy(self):
        index = 0
        for key in self.policy.keys():
            self.policy[key] = self.get_random_action(key)
        self.save_policy(f'random_policy_{FILE_NUM}.txt')
    
    def is_end_state(self, board, symbol):
        for i in range(BOARD_ROWS):
            #check row
            if symbol == board[i][0] == board[i][1] and board[i][1] == board[i][2]:
                return 1
            
            #check col
            if symbol == board[0][i] == board[1][i] and board[1][i] == board[2][i]:
                return 1

        #diagonal
        if symbol == board[0][0] == board[1][1] and board[1][1] == board[2][2]:
            return 1
        
        if symbol == board[2][0] == board[1][1] and board[1][1] == board[0][2]:
            return 1
        
        #draw
        if len(self.available_positions(board)) == 0:
            return 0.5

        return 0

    def save_policy(self, filename):
        print("Saving policy...")
        fw = open(filename, 'wb')
        pickle.dump(self.policy, fw)
        fw.close()
        print("Successfully saved policy!")
    
    def load_policy(self, filename):
        print("Initializing policy...")
        print("Loading policy...")
        fr = open(filename, 'rb')
        self.policy = pickle.load(fr)
        fr.close()
        print("Successfully loaded policy!")

    def save_states(self, filename):
        print("Saving states...")
        fw = open(filename, 'wb')
        pickle.dump(self.state_values, fw)
        fw.close()
        print("Successfully saved states!")

    #loads player states and updates policy
    def load_player_states(self):
        print("Initializing states...")
        print("Loading file...")
        fr = open("new_states_test.txt", 'rb')
        self.state_values = pickle.load(fr)
        self.policy = copy.deepcopy(self.state_values)
        fr.close()

        # #initialize random policy
        # self.update_policy()

        #read existing random policy
        self.load_policy('random_policy_2.txt')
        # self.load_policy('random_policy.txt')


        print("Successfully loaded states and policy!")

class QPlayer:
    def __init__(self):
        self.name = "yep"
    
    def a_function(self):
        pass


if __name__ == "__main__":
    teacher = Teacher() #X
    student = QPlayer() #O

    game = State(teacher, student)
    # game.initialize_player_states()
    # game.teacher.save_states('new_states_test.txt')

    start_datetime = datetime.now()
    start_time = time.time()
    game.play_game(max_policy_iter=FILE_NUM, max_value_iter=10000)
    end_datetime = datetime.now()
    print('Datetime: {}'.format(end_datetime - start_datetime))
    end_time = time.time() - start_time
    print(f"Time elapsed: {end_time} seconds")
