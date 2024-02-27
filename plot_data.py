from game_settings import SCORE_DATA_FILENAME
import matplotlib.pyplot as plt
import numpy as np


# Retrieve data from csv
scores = []
with open(SCORE_DATA_FILENAME) as file:
    # Skip first line
    x = file.readline()

    for line in file.readlines():
        score = int(line)
        scores.append(score)

# Plot
x = np.linspace(0, len(scores), len(scores))
y = np.array(scores)

plt.plot(x, y)
plt.show()


# # Load data from the first file
# scores = []
# with open('./data/latest.csv') as file:
#     next(file)  # Skip the header
#     for line in file:
#         score = int(line.strip())
#         scores.append(score)

# # Load data from the second file
# scores2 = []
# with open('./data/data_2layers128_64.csv') as file:
#     next(file)  # Skip the header
#     for line in file:
#         score = int(line.strip())
#         scores2.append(score)

# fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns of plots

# # Plot for the first file
# x = np.linspace(0, len(scores), len(scores))
# y = np.array(scores)
# axs[0].plot(x, y)
# axs[0].set_title('data_1layer256')

# # Plot for the second file
# x2 = np.linspace(0, len(scores2), len(scores2))
# y2 = np.array(scores2)
# axs[1].plot(x2, y2)
# axs[1].set_title('data_2layers128_64')

# plt.show()
