import numpy as np
import os

def create_sample_data():
    os.makedirs("data", exist_ok=True)

    t = np.linspace(0, 10, 100)
    y = 2 + 3*t + np.random.randn(len(t))*0.5
    A = np.column_stack([np.ones_like(t), t])
    data = np.column_stack([A, y])

    np.savetxt("data/example1.csv", data, delimiter=",")
    print("Saved data/example1.csv")

if __name__ == "__main__":
    create_sample_data()
