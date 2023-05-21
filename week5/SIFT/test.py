import matplotlib.pyplot as plt
import numpy as np
import torch

s1 = 64
s2 = 64
x = np.arange(-10,10,0.1)
x1 = torch.from_numpy(x)
y = torch.sigmoid(s1*x1)
y = y.cpu().numpy()

y2 = torch.tanh(s2*x1)
y2 = (y2+1.0) / 2
y2 = y2.cpu().numpy()


plt.figure()
plt.plot(x,y)
plt.plot(x,y2)

plt.show()
