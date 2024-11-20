import numpy as np

u = 1.0
t_0 = 0.0
t_end = 50.0
r = 1000.0
n_samples = 50
c = 0.01

def solution(t):
    return - t / (r*c) + np.log(u/r)

def get_dataset():
    t = np.linspace(t_0, t_end, n_samples)
    u = solution(t)
    return u,t


#if __name__ == '__main__':
    #print(get_dataset())