import scipy.io


def get_dataset():
    data = scipy.io.loadmat("data/burgers.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    return u_ref, t_star, x_star


u_ref, t_star, x_star = get_dataset()
u0 = u_ref[0, :]
print(u0)