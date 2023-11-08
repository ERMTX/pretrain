import torch
input = torch.load('/home/fu/argoverse2_forcast_mae/forecast-mae/val/scenario_00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff.pt')
print(input)
x = input['x']
print(x.shape)
x = x.cpu().detach().numpy()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)
for i in range(x.shape[0]):
    ax[0, 0].plot(x[i, :, 0], x[i, :, 1])

plt.show()