import numpy as np
import matplotlib.pyplot as plt


# data to plot
n_groups = 2
means_alu = (32.74, 15.87)
means_memory = (13.31, 37.46)


# create plot
fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.3
opacity = 0.8


rects1 = plt.bar(index, means_alu, bar_width,
alpha=opacity,
color='navy',
label='VectorALUBusy')


rects2 = plt.bar(index + bar_width, means_memory, bar_width,
alpha=opacity,
color='crimson',
label='MemUnitBusy')



plt.xlabel('Algorithm')
plt.ylabel('Time [%]')
# plt.title('')
plt.xticks(index + bar_width * 0.5, ('Binary Method', '4-bit Pippenger with CPU Reduction'))
plt.legend()


plt.tight_layout()
plt.savefig("profiler.png")