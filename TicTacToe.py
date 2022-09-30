import numpy as np
import random
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3

class State:
    def __init__(self, student, teacher):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board = self.board.astype(str)
        self.player1 = student
        self.player2 = teacher
        self.currentSymbol = None
        self.winningSymbol = None
        self.isEndGame = False

    def play_game(self, iterations):
        for i in range(iterations):
            print(f"Iteration: {i}")

            self.reset()
            self.player1.game_begin()
            self.player2.game_begin()
            # self.randomize_symbol()
            # print('player1', self.player1.symbol)
            # print('player2', self.player2.symbol)

            # isPlayer1 = random.choice([True, False])
            isX = random.choice([True, False])
            print(f"isx: {isX}")
            while not self.isEndGame:
                # if isPlayer1:
                #     move = self.player1.get_action(self.board, self.available_positions())
                #     self.currentSymbol = self.player1.symbol
                # else:
                #     move = self.player2.get_random_action(self.available_positions())
                #     self.currentSymbol = self.player2.symbol

                if isX:
                    move = self.player1.get_action(self.board, self.available_positions())
                else:
                    move = self.player2.get_random_action(self.available_positions())

                self.update_state(move, isX)
                game.display_board()

                reward, self.isEndGame, self.winningSymbol = self.check_win('X' if isX else 'O')
                print("reward: ", reward)
                if self.isEndGame:
                    current_state = self.get_hash()
                    if reward == 0.5:
                        self.player1.updateQ(reward, current_state, self.available_positions())
                        self.player2.updateQ(reward, current_state, self.available_positions())
                    elif reward == 0:
                        # if isPlayer1:
                        if isX:
                            self.player1.updateQ(reward, current_state, self.available_positions())
                        else:
                            self.player2.updateQ(reward, current_state, self.available_positions())
                    else:
                        # print("winner: ", self.currentSymbol)
                        print("winner: ", self.winningSymbol)
                        # if isPlayer1:
                        if self.winningSymbol == 'X' and isX:
                            self.player1.updateQ(reward, current_state, self.available_positions())
                            self.player2.updateQ(-1 * reward, current_state, self.available_positions())
                        else:
                            self.player1.updateQ(-1 * reward, current_state, self.available_positions())
                            self.player2.updateQ(reward, current_state, self.available_positions())
                    
                if (len(self.available_positions()) == 0):
                    self.isEndGame = True

                # isPlayer1 = not isPlayer1
                isX = not isX
        self.p1.save_policy()
    
    # for q-table key
    def get_hash(self):
        hash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
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
        if len(self.available_positions()) == 0:
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

    def available_positions(self):
        available = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i][j] == '0.0':
                    available.append((i,j))
        
        return available

    def update_state(self, position, isX):
        self.board[position[0]][position[1]] = 'X' if isX else 'O'

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board = self.board.astype(str)
        self.currentSymbol = None
        self.isEndGame = False
    
    def give_reward(self):
        pass


class QPlayer:
    def __init__(self, epsilon = 0.2, alpha=0.3, gamma=0.9):
        self.symbol = None
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # self.states_visited = []  
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
    
    #epsilon greedy get action
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

            # return max value be greedy) 
            if q_values.count(max_q) > 1:
                #gets indices of the max values in q_values
                actions_list = [i for i in range(len(possible_moves)) if q_values[i] == max_q]
                index = random.choice(actions_list)
            else:
                index = q_values.index(max_q)
            
            action = possible_moves[index]

        self.state_action_key = (state, action)
        self.state_action_value = self.getQ_value(state, action)
        return action
    
    def save_policy(self):
        fw = open('q_student_policy', 'wb')
        pickle.dump(self.q_table)
        fw.close()
    
    def load_policy(self, filename):
        fr = open(filename, 'rb')
        self.q_table = pickle.load(fr)
        fr.close()


    def printQ(self):
        print("===Q-table===")

        for i in self.q_table.keys():
            print(f"{i}: {self.q_table[i]}")
        
        print("=============")

    
    def updateQ(self, reward, state, available_positions):
        next_q_values = []
        for action in available_positions:
            next_q_values.append(self.getQ_value(state, action))
        
        max_Q_next = max(next_q_values) if next_q_values else 0.0

        print("Q(s,a) = Q(s,a) + alpha * ((reward + (y * maxQ(s', a'))) - Q(s,a))")
        print(f"Q(s,a) = {self.state_action_value} + {self.alpha} * (({reward} + ({self.gamma} * {max_Q_next})) - {self.state_action_value})")
        self.q_table[self.state_action_key] = self.state_action_value + (self.alpha * ((reward + (self.gamma * max_Q_next)) - self.state_action_value))
        print(f"new Q(s,a) = {self.q_table[self.state_action_key]}")
        self.printQ()


class RandomPlayer:
    def __init__(self, epsilon = 0.2, alpha=0.3, gamma=0.9):
        self.symbol = None
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_random_action(self, possible_moves):
        action = random.choice(possible_moves) ##action
        return action
    
    def get_action(self, possible_moves):
        #epsilon greedy
        action = random.choice(possible_moves) ##action
        return action
    
    def updateQ(self, reward, state, available_positions):
        pass

    def game_begin(self):
        pass


if __name__ == "__main__":
    protagonist = QPlayer() #protagonist
    antagonist = RandomPlayer() #antagonist, might have to have different functions for this one  
    game = State(protagonist, antagonist)
    game.display_board()
    # print(game.board.reshape(BOARD_ROWS * BOARD_COLS))
    # print(game.availablePositions())

    game.play_game(5)
    # protagonist.q_table[str(game.board.reshape(BOARD_ROWS * BOARD_COLS))] = 1
    # print(protagonist.q_table)