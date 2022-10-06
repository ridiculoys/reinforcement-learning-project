import pickle 
def load_file(filename):
    print("Loading file...")
    fr = open(filename, 'rb')
    q_table = pickle.load(fr)
    fr.close()
    print("Successfully loaded!")
    return q_table


random = load_file("against_random/new_q_student_policy_20000")
trained = load_file("against_trained/q_student_policy_20000")
# print('ra', random)
# print('tr', trained)
print(random == trained)