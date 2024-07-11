import numpy as np
import matplotlib.pyplot as plt

# Data
labels = ['Dimension1', 'Dimension2', 'Dimension3', 'Dimension4', 'Dimension5']
var1_values = [5, 4, 3, 2, 1]
var2_values = [3, 4, 5, 4, 3]

# Number of variables we're plotting.
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# The values to be plotted need to be complete the loop
var1_values += var1_values[:1]
var2_values += var2_values[:1]

# Create the figure and polar subplot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], labels)

# Draw ylabels
ax.set_rscale('linear')

# Plot data
ax.plot(angles, var1_values, linewidth=1, linestyle='solid', label='Variable 1')
ax.plot(angles, var2_values, linewidth=1, linestyle='solid', label='Variable 2')

# Fill area
ax.fill(angles, var1_values, 'b', alpha=0.1)
ax.fill(angles, var2_values, 'r', alpha=0.1)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()



#-------------
import numpy as np
import matplotlib.pyplot as plt

# Data
labels = ['Dimension1', 'Dimension2', 'Dimension3', 'Dimension4', 'Dimension5']
var1_values = [5, 400, 3000, 20, 1000]
var2_values = [3, 200, 2500, 40, 700]

# Normalize the data
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(value - min_val) / (max_val - min_val) for value in values]

var1_values_normalized = normalize(var1_values)
var2_values_normalized = normalize(var2_values)

# Number of variables we're plotting
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# The values to be plotted need to complete the loop
var1_values_normalized += var1_values_normalized[:1]
var2_values_normalized += var2_values_normalized[:1]

# Create the figure and polar subplot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], labels)

# Draw ylabels
ax.set_rscale('linear')

# Plot data
ax.plot(angles, var1_values_normalized, linewidth=1, linestyle='solid', label='Variable 1')
ax.plot(angles, var2_values_normalized, linewidth=1, linestyle='solid', label='Variable 2')

# Fill area
ax.fill(angles, var1_values_normalized, 'b', alpha=0.1)
ax.fill(angles, var2_values_normalized, 'r', alpha=0.1)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
