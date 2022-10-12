import numpy as np
import random
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
FILE_NUM = '30000'

class State:
    def __init__(self, student, teacher, policy = None, delta = None):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board = self.board.astype(str)
        self.player1 = student
        self.player2 = teacher
        self.currentSymbol = None
        self.winningSymbol = None
        self.isEndGame = False
        self.p1Wins = 0
        self.p1First = 0 #number of times p1 went first
        self.p2Wins = 0
        self.p2First = 0 #number of times p2 went first
        self.draws = 0

        # from : https://medium.com/@ngao7/markov-decision-process-policy-iteration-42d35ee87c82
        self.index = 0
        self.reward = None
        self.transition_prob = None
        self.gamma = None
        self.delta = delta
        self.policy = policy if policy != None else "random_policy()" #need to change this
        self.value_fx = None

    def play_game(self, max_policy_iter=10000, max_value_iter=10000):
        self.player1.initialize_states()

        for i in range(max_policy_iter):
            if i % 10 == 0:
                print(f"Iteration: {i}")

            #Initial assumption is that policy is stable
            optimal_policy_found = True

            for j in range(max_value_iter):
                max_diff = 0

                self.reset()
                self.player1.game_begin()
                self.player2.game_begin()

                isX = random.choice([True, False])
                print(f"isx: {isX}")
                self.update_first_data(isX)

                for s in self.player1.value_function.keys():
                    reward = self.get_reward(s)
                    for moves in available_positions:
                        val +=  probability * (self.player1.gamma * self.player1.state_values)



                
                while not self.isEndGame:
                    if isX:
                        move = self.player1.get_action(self.board, self.available_positions(self.board))
                        # move = self.player1.get_random_action(self.available_positions(self.board))
                    else:
                        move = self.player2.get_random_action(self.available_positions(self.board))
                        # move = self.player2.get_action(self.board, self.available_positions(self.board))

                    self.update_state(move, isX)
                    self.display_board()

                    reward, self.isEndGame, self.winningSymbol = self.check_win('X' if isX else 'O')
                    
                    

                    if self.isEndGame:
                        current_state = self.get_hash(self.board)
                        if reward == 0.5:
                            print("draw")
                            self.player1.updateQ(reward, current_state, self.available_positions(self.board))
                            self.player2.updateQ(reward, current_state, self.available_positions(self.board))
                            self.draws += 1
                        else:
                            print("winner: ", self.winningSymbol)
                            if self.winningSymbol == 'X' and isX:
                                self.player1.updateQ(reward, current_state, self.available_positions(self.board))
                                self.player2.updateQ(-1 * reward, current_state, self.available_positions(self.board))
                                self.p1Wins += 1
                            else:
                                self.player1.updateQ(-1 * reward, current_state, self.available_positions(self.board))
                                self.player2.updateQ(reward, current_state, self.available_positions(self.board))
                                self.p2Wins += 1

                    isX = not isX
    
    def play_game_2(self):
        isX = random.choice([True, False])
        print(f"isx: {isX}")

        while not self.isEndGame:
            if isX:
                move = self.player1.get_action(self.board, self.available_positions(self.board))
            else:
                move = self.player2.get_action(self.available_positions(self.board))

            self.update_state(move, isX)
            self.display_board()

            reward, self.isEndGame, self.winningSymbol = self.check_win('X' if isX else 'O')
            # print("reward: ", reward)
            if self.isEndGame:
                if reward == 0.5:
                    print(self.winningSymbol + "!")
                else:
                    print(self.winningSymbol + " wins!")

                cont = input("continue? y/n. ")
                if cont.lower() == "y":
                    self.reset()
                    self.display_board()
                else:
                    exit()

            isX = not isX


    # for q-table key
    def get_hash(self, board):
        # print(board)
        hash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return hash

    def randomize_symbol(self):
        symbol = random.randint(0,1)
        if symbol == 1: #player1 protag goes first goes first
            self.player1.symbol = 'X'
            self.player2.symbol = 'O'
        else: #player2 antag goes first
            self.player1.symbol = 'O'
            self.player2.symbol = 'X'

    def check_win(self, symbol):
        for i in range(BOARD_ROWS):
            #check row
            if symbol == self.board[i][0] == self.board[i][1] and self.board[i][1] == self.board[i][2]:
                return 1, True, symbol
            
            #check col
            if symbol == self.board[0][i] == self.board[1][i] and self.board[1][i] == self.board[2][i]:
                return 1, True, symbol

        #diagonal
        if symbol == self.board[0][0] == self.board[1][1] and self.board[1][1] == self.board[2][2]:
            return 1, True, symbol
        
        if symbol == self.board[2][0] == self.board[1][1] and self.board[1][1] == self.board[0][2]:
            return 1, True, symbol
        
        #draw
        if len(self.available_positions(self.board)) == 0:
            return 0.5, True, 'draw'
        
        return 0, False, 'no reward'

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

    def available_positions(self, board):
        available = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if board[i][j] == '0.0':
                    available.append((i,j))
        
        return available

    def update_state(self, position, isX):
        self.board[position[0]][position[1]] = 'X' if isX else 'O'

    def get_state(self, initial_state, turn):
        available_positions = self.available_positions(initial_state)
        if len(available_positions) == 0:
            hash = self.get_hash(initial_state)
            self.player1.state_values[hash] = 0
            return
        else:
            for move in available_positions:
                new_state = initial_state.copy()
                new_state[move] = turn
                hash = self.get_hash(new_state)
                self.get_state(new_state, 'X' if turn == 'O' else 'O')
                if hash in self.player1.state_values:
                    continue
                self.player1.state_values[hash] = 0
                self.index += 1
                print(self.index)

    def initialize_player_states(self):
        empty_hash = self.get_hash(self.board)
        self.player1.state_values[empty_hash] = 0
        self.index += 1
        print(self.index)
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                X_state = self.board.copy()
                X_state[i][j] = 'X'

                hash = self.get_hash(X_state)
                self.player1.state_values[hash] = 0
                self.index += 1
                print(self.index)

                ret = self.get_state(X_state, 'O')

        for k in range(BOARD_ROWS):
            for l in range(BOARD_COLS):
                O_state = self.board.copy()
                O_state[k][l] = 'O'

                hash = self.get_hash(O_state)
                self.player1.state_values[hash] = 0
                self.index += 1
                print(self.index)

                ret = self.get_state(O_state, 'X')




    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board = self.board.astype(str)
        self.currentSymbol = None
        self.isEndGame = False

    def update_first_data(self, isX):
        if isX:
            self.p1First += 1
        else:
            self.p2First += 1
    
    def save_data(self, filename):
        print("Saving win data...")
        fp = open(filename, "a")
        print(f"{FILE_NUM} {self.p1Wins} {self.p2Wins} {self.draws} {self.p1First} {self.p2First}\n")
        fp.write(f"{FILE_NUM} {self.p1Wins} {self.p2Wins} {self.draws} {self.p1First} {self.p2First}\n")
        fp.close()
        print("Successfully saved!")

class QPlayer:
    def __init__(self, trained = False, epsilon = 0.2, alpha=0.3, gamma=0.9):
        self.symbol = None
        self.trained = trained
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.state_action_key = None #to use for the calculation later
        self.state_action_value = None #to use for the calculation later

    def game_begin(self):
        self.state_action_key = None
        self.state_action_value = None

    # for q-table key
    def get_hash(self, board):
        hash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return hash

    def get_random_action(self, possible_moves):
        action = random.choice(possible_moves) ##action
        return action
    
    #gets current q value
    def getQ_value(self, state, action):
        q_value = self.q_table.get((state, action))

        if q_value is None:
            q_value = self.q_table[(state, action)] = 1.0
        return q_value
    
    #epsilon greedy policy get action
    def get_action(self, board, possible_moves):
        state = self.get_hash(board)
        if random.random() < self.epsilon:
            action = random.choice(possible_moves) ##action
        else:
            #get q values for next possible moves, get max
            q_values = []
            for action in possible_moves:
                q_values.append(self.getQ_value(state, action))
            max_q = max(q_values)

            print('q', q_values)

            # return max value be greedy) 
            if q_values.count(max_q) > 1:
                #gets indices of the max values in q_values
                actions_list = [i for i in range(len(possible_moves)) if q_values[i] == max_q]
                print('actions', actions_list)
                index = random.choice(actions_list)
            else:
                index = q_values.index(max_q)
            
            print('index', index)
            
            action = possible_moves[index]
            print('possible', possible_moves)
            print('poss[index]', action)

        self.state_action_key = (state, action)
        self.state_action_value = self.getQ_value(state, action)
        return action
    
    def save_policy(self, filename):
        print("Saving policy...")
        fw = open(filename, 'wb')
        pickle.dump(self.q_table, fw)
        fw.close()
        print("Successfully saved!")
    
    def load_policy(self, filename):
        print("Loading policy...")
        fr = open(filename, 'rb')
        self.q_table = pickle.load(fr)
        fr.close()
        print("Successfully loaded!")


    def printQ(self):
        print("===Q-table===")
        for i in self.q_table.keys():
            print(f"{i}: {self.q_table[i]}")
        
        print("=============")

    
    def updateQ(self, reward, state, available_positions):
        if not self.trained:
            next_q_values = []
            for action in available_positions:
                next_q_values.append(self.getQ_value(state, action))
            
            max_Q_next = max(next_q_values) if next_q_values else 0.0

            print("Q(s,a) = Q(s,a) + alpha * ((reward + (y * maxQ(s', a'))) - Q(s,a))")
            print(f"Q(s,a) = {self.state_action_value} + {self.alpha} * (({reward} + ({self.gamma} * {max_Q_next})) - {self.state_action_value})")
            self.q_table[self.state_action_key] = self.state_action_value + (self.alpha * ((reward + (self.gamma * max_Q_next)) - self.state_action_value))
            print(f"new Q(s,a) = {self.q_table[self.state_action_key]}")
            # self.printQ()

class Agent:
    def __init__(self, epsilon = 0.2, gamma=0.9, delta=10):
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.state_values = {}
        self.policy = {}
        self.q_table = {}

    def initialize_player_states(self):
        print("Initializing states...")
        print("Loading file...")
        fr = open("states.txt", 'rb')
        self.state_values = pickle.load(fr)
        fr.close()
        print("Successfully loaded!")

    #epsilon greedy policy get action
    def get_action(self, board, possible_moves):
        state = self.get_hash(board)
        if random.random() < self.epsilon:
            action = random.choice(possible_moves) ##action
        else:
            #get q values for next possible moves, get max
            q_values = []
            for action in possible_moves:
                q_values.append(self.getQ_value(state, action))
            max_q = max(q_values)

            print('q', q_values)

            # return max value be greedy) 
            if q_values.count(max_q) > 1:
                #gets indices of the max values in q_values
                actions_list = [i for i in range(len(possible_moves)) if q_values[i] == max_q]
                print('actions', actions_list)
                index = random.choice(actions_list)
            else:
                index = q_values.index(max_q)
            
            print('index', index)
            
            action = possible_moves[index]
            print('possible', possible_moves)
            print('poss[index]', action)

        self.state_action_key = (state, action)
        self.state_action_value = self.getQ_value(state, action)
        return action
    
    def save_states(self, filename):
        print("Saving states...")
        fw = open(filename, 'wb')
        pickle.dump(self.state_values, fw)
        fw.close()
        print("Successfully saved!")
    
    def load_states(self, filename):
        print("Loading states...")
        fr = open(filename, 'rb')
        self.state_values = pickle.load(fr)
        fr.close()
        print("Successfully loaded!")
        

class RandomPlayer:
    def __init__(self, epsilon = 0.2, alpha=0.3, gamma=0.9):
        self.symbol = None
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_random_action(self, available_positions):
        action = random.choice(available_positions) ##action
        return action
    
    def get_action(self, available_positions):
        #epsilon greedy
        action = random.choice(available_positions) ##action
        return action
    
    def updateQ(self, reward, state, available_positions):
        pass

    def game_begin(self):
        pass

    def save_policy(self):
        pass

#code th
class HumanPlayer:
    def __init__(self):
        self.name = "hooman"
    
    def get_action(self, available_positions):
        while True:
            row = int(input("Input row: "))
            col = int(input("Input col: "))
            action = (row, col)
            if action in available_positions:
                return action
    
    def updateQ(self, reward, state, available_positions):
        pass

    def game_begin(self):
        pass


if __name__ == "__main__":
    # random versus random
    protagonist = Agent() #teacher
    # protagonist.initialize_player_states()
    antagonist = RandomPlayer() #trained q-learning
    game = State(protagonist, antagonist)
    # game.display_board()
    
    # game.initialize_player_states()
    # print(len(game.player1.state_values))
    # game.player1.save_states("states.txt")
    
    # game.play_game(int(FILE_NUM))
    # game.play_game(max_policy_iter=10, max_value_iter=10)
