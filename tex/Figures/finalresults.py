import numpy as np
import matplotlib.pyplot as plt


# data to plot
n_groups = 6
means_i7 = (0.75, 0, 0, 0, 0, 0)
means_arm = (0, 1.38, 0, 0, 0, 0)
means_binary= (0, 0, 6, 6, 5, 4)
means_window = (0, 0, 6, 6, 3, 2.5)
means_4pip = (0, 0, 6, 6, 2.5, 1.72)
means_1pip = (0, 0, 6, 6, 1.8, 1.05)
means_4pipcpu = (0, 0, 6, 6, 1.6, 0.96)


# create plot
fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.3
opacity = 1.0

print(index)
index = index * 10 * bar_width
plt.gca().yaxis.grid(True)
plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(color='gray', linestyle='dashed')

rects1 = plt.bar(index, means_i7, bar_width,
alpha=opacity,
color='navy',
label='Intel i7 Multiexp')


rects2 = plt.bar(index + bar_width, means_arm, bar_width,
alpha=opacity,
color='darkgreen',
label='ARM Multiexp')


rects3 = plt.bar(index + 2*bar_width, means_binary, bar_width,
alpha=opacity,
color='crimson',
label='Binary Method')

rects4 = plt.bar(index + 3*bar_width, means_window, bar_width,
alpha=opacity,
color='violet',
label='Sliding Window')

rects5 = plt.bar(index + 4*bar_width, means_4pip, bar_width,
alpha=opacity,
color='orange',
label='4-bit Pippenger')

rects6 = plt.bar(index + 5*bar_width, means_1pip, bar_width,
alpha=opacity,
color='dodgerblue',
label='1-bit Pippenger')

rects7 = plt.bar(index + 6*bar_width, means_4pipcpu, bar_width,
alpha=opacity,
color='seagreen',
label='4-bit Pippenger with CPU Reduction')


plt.xlabel('Devices')
plt.ylabel('Time [s]')
# plt.title('')
ticks = index + 4*bar_width
ticks[0] = index[0]
ticks[1] = index[1] + bar_width
plt.xticks(ticks, ('Intel 7700HQ', 'ARM Exynos 9810', 'Mali G72', 'Intel HD Graphics 630', 'NVIDIA 1060', 'AMD RX 580'))

plt.legend()

print(index)

plt.tight_layout()

plt.show()
#plt.savefig("finalresults.png")