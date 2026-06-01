import csv

import numpy as np

def simulate_datasets (X, Y ,scale):
    return X*scale, Y*scale

def main():
    #read in X_1 /Y_1
    with open("Tierdaten/X_anim1.csv", "r") as f:
        X_1 = csv.reader(f)
        X_1 = np.array([float(row[0]) for row in X_1])
        print(X_1)

    with open("Tierdaten/Y_anim1.csv", "r") as f:
        Y_1 = csv.reader(f)
        Y_1 = np.array([float(row[0]) for row in Y_1])
        print(Y_1)

    # simulate data with different scales
    scales = [0.5,0.8,2.35,4.67,0.24,1.89]

    X_2_sim, Y_2_sim = simulate_datasets(X_1, Y_1, scales[0])
    X_3_sim, Y_3_sim = simulate_datasets(X_1, Y_1, scales[1])
    X_4_sim, Y_4_sim = simulate_datasets(X_1, Y_1, scales[2])
    X_5_sim, Y_5_sim = simulate_datasets(X_1, Y_1, scales[3])
    X_6_sim, Y_6_sim = simulate_datasets(X_1, Y_1, scales[4])
    X_7_sim, Y_7_sim = simulate_datasets(X_1, Y_1, scales[5])

    print(Y_7_sim)
    print(X_7_sim)

if __name__ == "__main__":
    main()



