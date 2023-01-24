import pickle
FILE_NUM = 30000

def return_states(table):
    return len(table.keys())

def load_policy(filename):
    print("Loading policy...")
    fr = open(filename, 'rb')
    table = pickle.load(fr)
    fr.close()
    print("Successfully loaded!")
    return table

# table = load_policy(f"against_random/new_q_student_policy_{FILE_NUM}")
# table = load_policy(f"against_trained/q_student_policy_{FILE_NUM}")
table = load_policy(f"test_q_student_policy_{FILE_NUM}")
print(return_states(table))