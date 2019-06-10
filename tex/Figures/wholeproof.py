import numpy as np
import matplotlib.pyplot as plt


# data to plot
n_groups = 2
means_64 = (2, 2.95)
means_64_stress = (2.3, 4.10)
means_32= (8.5, 18.6)


# create plot
fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.3
opacity = 0.8


rects1 = plt.bar(index, means_64, bar_width,
alpha=opacity,
color='b',
label='64-bit')


rects2 = plt.bar(index + bar_width, means_64_stress, bar_width,
alpha=opacity,
color='g',
label='64-bit Stress Test')


rects2 = plt.bar(index + 2*bar_width, means_32, bar_width,
alpha=opacity,
color='r',
label='32-bit')


plt.xlabel('Architecture')
plt.ylabel('Time [s]')
# plt.title('')
plt.xticks(index + bar_width, ('x86', 'ARM'))
plt.legend()


plt.tight_layout()
plt.savefig("wholeproof.png", dpi=300)