import scipy.io


def get_dataset():
    data = scipy.io.loadmat("data/burgers.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()

    return u_ref, t_star
