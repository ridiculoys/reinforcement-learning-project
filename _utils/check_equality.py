import pickle 
def load_file(filename):
    print("Loading file...")
    fr = open(filename, 'rb')
    q_table = pickle.load(fr)
    fr.close()
    print("Successfully loaded!")
    return q_table

def print_dict(name, dic):
    print(name)
    count = 0
    for key,value in dic.items():
        print(f"key: {key}, value: {value}")
        print()
        count += 1
        if count > 10:
            break


# random = load_file("against_random/new_q_student_policy_20000")
# trained = load_file("against_trained/q_student_policy_20000")
random = load_file("random_policy.txt")
trained = load_file("new_trained_policy.txt")
print_dict('RANDOM', random)
print_dict('TRAINED', trained)
# print(random == trained)