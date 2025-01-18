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

# Required Arguments
# Optional Arguments
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

# Workflow

## Input File Reading:
Reads the input dataset from a space-separated text file.
Dynamically assigns column names as label + specified coordinate columns.

## Offset Point Calculation:
Calculates the centroid of the point cloud.
Uses PCA to find the principal axes.
Offsets points outward using a combination of the outward vector and weighted PCA axes.

## Output File Writing:
Saves the updated dataset, including new offset points, to the specified output file.

# Error Handling

Missing Input File: Logs an error and exits if the file is not found.
Invalid Parameters: Validates all inputs and raises appropriate exceptions for invalid values.
Processing Errors: Logs errors for individual points but continues processing the rest.

# Extensibility

Support for Higher Dimensions: Modify --point-cols to handle additional dimensions.
Adjust Offset Behavior: Customize the cutoff function to change how the offset is calculated.
Dynamic Labels: Use custom labels for both input and output points.

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
