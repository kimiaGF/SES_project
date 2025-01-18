# Offset Points Generator

This script processes a 3D dataset to generate new offset points for specified labeled data points. It calculates outward offsets based on a given offset magnitude and appends the new points to the dataset.

## Overview

The script identifies points with a specific label in a 3D dataset, calculates their offset in a direction pointing outward from the centroid of the point cloud, and saves the updated dataset to an output file. The offset direction is determined using PCA and a weighted cutoff function.

## Running script
The script can be run from the commandline:
```bash
python script.py 
```

or with a Docker image
```bash
docker run --rm -v "$(pwd)/.:/app/." offset-points-generator
```

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
- `matplotlib`

Install dependencies using:
```bash
pip install numpy pandas scikit-learn matplotlib
```
# Workflow

## Input File Reading:
Reads the input dataset from a space-separated text file. Dataset is stored as a `pandas` DataFrame with columns: `label`,`x`,`y`,`z`.
For larger dimensionality data, an optional argument `point_cols` can be defined to label features accordingly.

## Offset Point Calculation:
Calculates the outward vector for each point "B" by using the centroid of the point cloud.

$$
\hat v_{outward} = \frac{r_{B} - r_{centroid}}{|r_{B} - r_{centroid}|}
$$

here, 
- $r_{B}$ is the coordinates of a "B" point of interest
- $r_{centroid}$ is the coordinates of the centroid of all data (computed using the average of all coordinates).

To account for local geometry variability, Principal Component Analysis (PCA) is conducted for each point "B" to find the axes of maximum variability. The number of components used for PCA is the same as data dimensionality (i.e. 3 for 3D data).
The offset direction is determined with a weighted combination of the outward vector and weighted PCA axes. 

$$
d_{offset} = \hat v_{outward} + \begin{bmatrix}w_1 \\ w_2 \\ \dots \\ w_n \end{bmatrix} \cdot \begin{bmatrix}\textbf{u}_1 \\ \textbf{u}_2 \\ \dots \\ \textbf{u}_n \end{bmatrix}
$$

and normalized to get a unit vector...


$$
\hat d_{offset} = \frac{d_{offset}}{|d_{offset}|}
$$


here, 
- $\textbf{u}_1, \dots, \textbf{u}_n$ are the principal axes determined by PCA. The number of components $n$ is the same as data dimensionality.
- $w_1,\dots,w_n$ are the weights assigned to each principal component axis.

The weights assigned to each principal component axis follow an exponential decay toward 0.

$$
w = \alpha * e^{-nx}
$$

here,
- $\alpha$ controls the weight assigned to the first principal component.
- $n$ is the dimensionality of the data and controls the rate of decay to 0.
  
![Weight function](plots/weight_function.png)

## Output File Writing:
Saves the updated dataset, including new offset points, to the specified output file.

![Side-by-Side 3D Scatter Plots](plots/3d_scatter_plots.png "Side-by-Side 3D Scatter Plots")

Interactive plots:
[Interactive Plotly Chart Original Dataset](plots/original_3d_figure.html)
[Interactive Plotly Chart Updated Dataset](plots/updated_3d_figure.html)


# Error Handling

Missing Input File: Logs an error and exits if the file is not found.
Invalid Parameters: Validates all inputs and raises appropriate exceptions for invalid values.
Processing Errors: Logs errors for individual points but continues processing the rest.

# Extensibility

Support for Higher Dimensions: Modify --point-cols to handle additional dimensions.
Adjust Offset Behavior: Customize the cutoff function to change how the offset is calculated.
Dynamic Labels: Use custom labels for both input and output points.

# Input File Format

The input file must be space-separated and contain:

A label column (indicating the label of each point).
Coordinate columns for the 3D space (default: x, y, z).
Example Input File (input.txt):
```
A 1.0 2.0 3.0
B 4.0 5.0 6.0
A 7.0 8.0 9.0
B 10.0 11.0 12.0
```
# Output File Format

The output file is a tab-separated text file containing:

Original points from the dataset.
Newly calculated offset points with an updated label (default: C).
Example Output File (output.txt):
```
label    x     y     z
A        1.0   2.0   3.0
B        4.0   5.0   6.0
A        7.0   8.0   9.0
B        10.0  11.0  12.0
C        5.2   6.3   7.1
C        11.5  12.6  13.2
```
# Command-Line Arguments

Run the script with the following command-line options:

## Optional Arguments
- `-i` or `--input`: Path to the input text file. (Default: `cdd.txt`)
- `-d` or `--offset-magnitude`: Magnitude of the offset. (Default: `2.0`)
- `-o` or `--output`: Path to the output text file. (Default: `out.txt`)
- `-a` or `--alpha`: Alpha value for the cutoff function. (Default: `0.5`)
- `--point-cols`: List of column names representing the coordinates. (Default: `['x', 'y', 'z']`)
- `-l` or `--label`: Label of the points to offset. (Default: `'B'`)
- `--offset-label`: Label to assign to the offset points. (Default: `'C'`)
## Example Command
```bash
python script.py -i input.txt -o output.txt -d 3.0 -a 0.5 --point-cols x y z -l B --offset-label C
```
# Example Workflow

## Input File (`cdd.txt`):
```
A 1.0 2.0 3.0
B 4.0 5.0 6.0
A 7.0 8.0 9.0
B 10.0 11.0 12.0
```
## Command:

`python script.py -i input.txt -o output.txt -d 3.0 -a 0.5 --point-cols x y z -l B --offset-label C`

## Output File (`out.txt`):
```
label    x     y     z
A        1.0   2.0   3.0
B        4.0   5.0   6.0
A        7.0   8.0   9.0
B        10.0  11.0  12.0
C        5.2   6.3   7.1
C        11.5  12.6  13.2
```
### License

This script is open-source and can be freely modified or extended for your use case.


This is a complete markdown documentation covering every aspect of the script
