"""
DA - LAB 1 - DA24C026

Data Loading
"""

import pandas as pd

dataset = pd.read_csv("Default_Dataset.csv", header = None)

dataset.head()

"""Data Plotting"""

# Plotting Data Points to check and clean inappropiate coordinates
import matplotlib.pyplot as plt
fig, axes=plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(w=10, h=10)

axes[0,0].scatter(dataset[0],dataset[1])
axes[0,0].set_xlabel("x")
axes[0,0].set_ylabel("y")
axes[0,0].set_title("Original Image - Scatter Plot")

"""Data Transformation to Sparse Matrix"""

dataset = dataset.dropna()  # Remove rows with missing values

import numpy as np

# Initializing a 1000x1000 matrix with zeros
matrix_size = 1000
matrix = np.zeros((matrix_size, matrix_size), dtype=bool)

# Mapped coordinates to matrix indices
for _, row in dataset.iterrows():
    x = round(row[0] / 100 * matrix_size)  # Scaling to 1000x1000
    y = round(row[1] / 100 * matrix_size)
    matrix[x, y] = True  # Mark the pixel as True for a red point

row,col = np.nonzero(matrix) # Discrete Datapoints of Original Image

axes[0,1].scatter(row,col,s=3, marker = 'o')
axes[0,1].set_xlabel("x")
axes[0,1].set_ylabel("y")
axes[0,1].set_title("Sparse Matrix - Scatter Plot")

"""Transformation Matrix for Horizontal Flip"""

matrix_flip_trans = np.zeros((1000, 1000), dtype=int)

# Setting the Alternate diagonal to 1
for i in range(1000):
    matrix_flip_trans[999-i, i] = 1

flipped_matrix = np.dot(matrix, matrix_flip_trans) # Horizontally flipped Matrix

rows,cols = np.nonzero(flipped_matrix) # Conversion from Sparse to Dense datapoints

axes[1,0].scatter(rows,cols,s=3, marker = 'o')
axes[1,0].set_xlabel("x")
axes[1,0].set_ylabel("y")
axes[1,0].set_title("Horizontally Flipped Image")

"""Transformation Matrix for 90 degree rotation"""

transpose_mat = np.transpose(matrix) # Matrix Transpose

rotated_matrix = np.dot(transpose_mat, matrix_flip_trans) # 90 Degree rotation by applying flip transformation on Transposed image

row1,col1 = np.nonzero(rotated_matrix) # Dense Discrete Datapoints of Rotated Image

axes[1,1].scatter(row1 ,col1 ,s=3, marker = 'o')
axes[1,1].set_xlabel("x")
axes[1,1].set_ylabel("y")
axes[1,1].set_title("Rotation by 90 Degrees")

fig.tight_layout()
plt.show()