import numpy as np
import matplotlib.pyplot as plt


# data to plot
n_groups = 1
means_64 = (0.3)
means_64_stress = (1.65)


# create plot
fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.3
opacity = 0.8


rects1 = plt.bar(index, means_64, bar_width,
alpha=opacity,
color='b',
label='FFT')


rects2 = plt.bar(index + 1.5*bar_width, means_64_stress, bar_width,
alpha=opacity,
color='g',
label='Multiexponentiation')


#plt.xlabel('Architecture')
plt.ylabel('Time [s]')
# plt.title('')
plt.xticks((index, index + 1.5*bar_width), ('FFT', 'Multiexponentiation'))
#plt.legend()


plt.tight_layout()
plt.savefig("proofparts.png")