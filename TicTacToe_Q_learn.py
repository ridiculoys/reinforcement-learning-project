import numpy as np
import random
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
FILE_NUM = '30000'

class State:
    def __init__(self, student, teacher):
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

    def play_game(self, iterations):
        for i in range(iterations):
            # if i % 10 == 0:
            #     print(f"Iteration: {i}")

            self.reset()
            self.player1.game_begin()
            self.player2.game_begin()

            isX = random.choice([True, False])
            print(f"isx: {isX}")
            self.update_first_data(isX)
            
            while not self.isEndGame:
                if isX:
                    move = self.player1.get_action(self.board, self.available_positions())
                    # move = self.player1.get_random_action(self.available_positions())
                else:
                    # move = self.player2.get_random_action(self.available_positions())
                    move = self.player2.get_action(self.board, self.available_positions())

                self.update_state(move, isX)
                self.display_board()

                reward, self.isEndGame, self.winningSymbol = self.check_win('X' if isX else 'O')
                if self.isEndGame:
                    current_state = self.get_hash()
                    if reward == 0.5:
                        print("draw")
                        self.player1.updateQ(reward, current_state, self.available_positions())
                        self.player2.updateQ(reward, current_state, self.available_positions())
                        self.draws += 1
                    else:
                        print("winner: ", self.winningSymbol)
                        if self.winningSymbol == 'X' and isX:
                            self.player1.updateQ(reward, current_state, self.available_positions())
                            self.player2.updateQ(-1 * reward, current_state, self.available_positions())
                            self.p1Wins += 1
                        else:
                            self.player1.updateQ(-1 * reward, current_state, self.available_positions())
                            self.player2.updateQ(reward, current_state, self.available_positions())
                            self.p2Wins += 1

                isX = not isX
        # self.player1.save_policy(f"against_random/new_q_student_policy_{FILE_NUM}")
        # self.player2.save_policy(f"player_2_save/q_student_policy_{FILE_NUM}")
        # self.save_data("1_random_v_random.txt")
        # self.save_data("2_q_v_random.txt")
        # self.save_data("3_trained_v_random.txt")
        # self.save_data("4_1_trained_v_q.txt")
        self.save_data("5_trained_v_trained.txt")
    
    def play_game_2(self):
        isX = random.choice([True, False])
        print(f"isx: {isX}")

        while not self.isEndGame:
            if isX:
                move = self.player1.get_action(self.board, self.available_positions())
                # move = self.player1.get_action(self.available_positions())
            else:
                move = self.player2.get_action(self.available_positions())
                # move = self.player2.get_action(self.board, self.available_positions())


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

#policy_iteration
class Teacher:
    def __init__(self):
        self.name = "policy"
        self.policy = {}

    def get_hash(self, board):
        hash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return hash
    
    def get_action(self, board, available_positions):
        hash = self.get_hash(board)

        if hash in self.policy.keys():
            return self.policy[hash]

        return random.choice(available_positions)
    
    def updateQ(self, reward, state, available_positions):
        pass

    def game_begin(self):
        pass

    def load_policy(self, filename):
        print("Initializing policy...")
        print("Loading policy...")
        fr = open(filename, 'rb')
        self.policy = pickle.load(fr)
        fr.close()
        print("Successfully loaded policy!")

if __name__ == "__main__":
    # random versus random
    # protagonist = RandomPlayer() #protagonist
    # antagonist = RandomPlayer() #antagonist, might have to have different functions for this one  
    # game = State(protagonist, antagonist)
    # game.display_board()
    # game.play_game(int(FILE_NUM))

    #Q-learn versus random
    # protagonist = QPlayer() #protagonist
    # antagonist = RandomPlayer() #antagonist, might have to have different functions for this one  
    # game = State(protagonist, antagonist)
    # game.display_board()
    # game.play_game(int(FILE_NUM))

    # trained versus random
    # protagonist = QPlayer(trained=True, epsilon=0) #protagonist
    # protagonist.load_policy(f"against_random/new_q_student_policy_{FILE_NUM}")
    # antagonist = RandomPlayer() #antagonist, might have to have different functions for this one
    # game = State(protagonist, antagonist)
    # game.display_board()
    # game.play_game(int(FILE_NUM))

    #Q-learn versus trained
    # protagonist = QPlayer() # trained to use O, p2
    # antagonist = QPlayer(trained=True, epsilon=0) #trained to use X, so p1
    # antagonist.load_policy(f"against_random/new_q_student_policy_{FILE_NUM}")
    # game = State(antagonist, protagonist)
    # game.display_board()
    # game.play_game(int(FILE_NUM))

    #trained versus trained
    # protagonist = QPlayer(trained=True, epsilon=0)
    # protagonist.load_policy(f"against_random/new_q_student_policy_{FILE_NUM}")
    # antagonist = QPlayer(trained=True, epsilon=0)
    # antagonist.load_policy(f"player_2_save/q_student_policy_{FILE_NUM}")
    # game = State(protagonist, antagonist)
    # game.display_board()
    # game.play_game(int(FILE_NUM))

    # trained Q versus hooman
    # protagonist = QPlayer(trained=True, epsilon=0)
    # protagonist.load_policy(f"against_random/new_q_student_policy_{FILE_NUM}")
    # protagonist.load_policy(f"player_2_save/q_student_policy_{FILE_NUM}")
    # hooman = HumanPlayer()
    # game = State(protagonist, hooman)
    # game = State(protagonist, hooman)
    # game = State(hooman, protagonist)
    # game.display_board()
    # game.play_game_2()

    # trained Q versus hooman
    protagonist = Teacher()
    protagonist.load_policy("teacher_trained_policy.txt")
    hooman = HumanPlayer()
    game = State(protagonist, hooman)
    game.display_board()
    game.play_game_2()
