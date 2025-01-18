# Offset Points Generator

This script processes a 3D dataset to generate new offset points for specified labeled data points. It calculates outward offsets based on a given offset magnitude and appends the new points to the dataset.

## Overview

The script identifies points with a specific label in a 3D dataset, calculates their offset in a direction pointing outward from the centroid of the point cloud, and saves the updated dataset to an output file. The offset direction is determined using PCA and a weighted cutoff function.

## Features

- **Dynamic Input and Output**: Specify input and output file paths.
- **Customizable Offset Parameters**: Control offset magnitude and alpha for smooth cutoff transitions.
- **Support for Different Dimensions**: Handle 3D or higher-dimensional datasets by specifying coordinate columns.
- **Error Handling**: Robust logging and validation to handle invalid inputs or processing errors.

---

## Dependencies

Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `argparse`

Install dependencies using:
```bash
pip install numpy pandas scikit-learn
