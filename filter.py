import h5py
import numpy as np
import matplotlib.pyplot as plt

'''
hf = h5py.File('/home/saitama/Desktop/Magazine/data.h5', 'r')
data = np.array(hf.get('dataset_1'))

plt.plot(data[10000:14000])
plt.show()
'''

alpha = 0.1
samples = 5000
frequency = 10

n = np.linspace(0, 1, samples)
noise = np.random.randn(samples)*0.09
x = np.sin(2*np.pi*frequency*n) + noise
y = np.zeros(samples)
y_simple = np.zeros(samples)


for i in range(1, samples):
    y[i] = alpha*x[i] + (1-alpha)*y[i-1]

for i in range(1, samples):
    y_simple[i] = (x[i] + x[i-1])*0.5


fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0, 0].set_title("Original Signal")
axs[0, 0].plot(n, np.sin(2*np.pi*frequency*n), 'red')
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Amplitude")

axs[0, 1].set_title("Noisy Signal")
axs[0, 1].plot(n, x, 'green')
axs[0, 1].set_xlabel("Time")
axs[0, 1].set_ylabel("Amplitude")

axs[1, 0].set_title("Simple LPF Output Signal")
axs[1, 0].plot(n, y_simple, 'orange')
axs[1, 0].set_xlabel("Time")
axs[1, 0].set_ylabel("Amplitude")

axs[1, 1].set_title("Better LPF Output Signal")
axs[1, 1].plot(n, y)
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("Amplitude")
plt.tight_layout()
plt.savefig("/home/saitama/Desktop/Magazine/plot.png")
