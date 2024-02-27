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
