import re
import numpy as np
import matplotlib.pyplot as plt

# Read file
with open("/Users/issraali/Documents/ipp/git/simsopt/examples/3_Advanced/spline_bayesian_opt/out_free_ls.txt", "r") as f:
    text = f.read()

# Regex to capture the tensor contents inside Length scales
pattern = r"Length scales:\s*tensor\(\[\[(.*?)\]\],\s*dtype=torch\.float64\)"
matches = re.findall(pattern, text, re.DOTALL)

length_scales = []

for m in matches:
    # Remove newlines and split numbers
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", m)
    nums = [float(x) for x in nums]
    length_scales.append(nums)

length_scales = np.array(length_scales)

print("Shape:", length_scales.shape)  # (iterations, dimensions)

# Plot
for i in range(length_scales.shape[1]):
    plt.plot(length_scales[:, i])

plt.xlabel("Iteration")
plt.ylabel("Length scale value")
plt.title("Length scales over iterations")

plt.show()