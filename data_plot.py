import pandas as pd

def load_data(filename):
    fp = open(filename, "r")
    data = [[j.strip() for j in i.split(" ")] for i in fp]
    df =  pd.DataFrame(data=data, columns=["iter", "p1_win", "p2_win", "draw", "p1_start", "p2_start"])
    fp.close()

    return df

random_v_random = load_data("1_random_v_random.txt")
q_v_random = load_data("2_q_v_random.txt")
trained_v_random = load_data("3_trained_v_random.txt")
q_v_trained = load_data("4_q_v_trained.txt")

data = [random_v_random, q_v_random, trained_v_random, q_v_trained]
for df in data:
    print(df)