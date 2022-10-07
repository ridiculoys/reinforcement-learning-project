import pandas as pd

def load_data(filename):
    fp = open(filename, "r")
    data = [[j.strip() for j in i.split(" ")] for i in fp]
    df =  pd.DataFrame(data=data, columns=["iter", "p1_win", "p2_win", "draw", "p1_start", "p2_start"])
    fp.close()

    return df

def plot_data(title, df):
    data = pd.DataFrame({
        'p1': df.loc[:, "p1_win"].tolist(),
        'p2': df.loc[:, "p2_win"].tolist(),
        'draw': df.loc[:, "draw"].tolist()
    }, index=df.loc[:, "iter"].tolist())

    print(data)

    lines = data.plot.line()



random_v_random = load_data("1_random_v_random.txt")
q_v_random = load_data("2_q_v_random.txt")
trained_v_random = load_data("3_trained_v_random.txt")
q_v_trained = load_data("4_q_v_trained.txt")

data = [random_v_random, q_v_random, trained_v_random, q_v_trained]
for df in data:
    print(df)

    # plot_data("Random vs Random", random_v_random)

df = pd.DataFrame({
   'pig': [20, 18, 489, 675, 1776],
   'horse': [4, 25, 281, 600, 1900]
   }, index=[1990, 1997, 2003, 2009, 2014])
lines = df.plot.line()