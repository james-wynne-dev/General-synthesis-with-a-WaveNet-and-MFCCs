import numpy as np
import matplotlib.pyplot as plt


data1 = "./Data/GraphData/Exp12_loss.npy"
data1 = np.load(data1)

# data2 = "./Data/GraphData/Exp14_loss.npy"
# data2 = np.load(data2)

# data3 = "./Data/GraphData/Exp8_loss.npy"
# data3 = np.load(data3)

def plot(data, label, window, alpha, color):
    x_vals = data[:,0]
    y_vals = data[:,1]
    if window:
        y_vals = smooth(y_vals, window)
    plt.plot(x_vals, y_vals, label=label, alpha=alpha, color=color)
    plt.ylabel("Loss")
    plt.xlabel("Steps")

def smooth(scalars, window):
    y_vals = np.zeros([scalars.shape[0]])
    half_window = int(window/2)
    for i in range(len(scalars)):
        start = max(0, i - half_window)
        end = min(scalars.shape[0], i + half_window)
        piece = scalars[start:end]
        size = piece.shape[0]
        window = np.ones([size]) / size
        y_vals[i] = sum(piece * window)
    return y_vals


plt.title("Experiment 12")
plot(data1, "Exp 12 - Moving Av.", 30, alpha=1, color="blue")
plot(data1, "Exp. 12", None, alpha=0.2, color="blue")
# plot(data2, "Exp 10 - Moving Av.", 30, alpha=1, color="blue")
# plot(data2, "Exp 10", None, alpha=1, color="blue")
# plot(data3, "Exp 8 - Moving Av.", 30, alpha=1, color="red")
# plot(data3, "Exp 8", None, alpha=0.2, color="red")
# plt.xlim(0,50000)
# plt.ylim(0,10)
plt.legend()


plt.show()
