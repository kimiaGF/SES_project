
# **Script Documentation**

## **Overview**
This script processes a 3D dataset to add offset points to a subset of data based on a specified label. The offset is applied in an outward direction determined by the principal components of the data. The script provides options for visualizing the results and saving the output to a file.

---

## **Features**
1. **Cutoff Function**:
   - Smoothly transitions a value to 0 using an exponential decay with adjustable degrees.
2. **Add Offset Points**:
   - Creates new points offset from the labeled points based on PCA and outward vectors.
3. **Visualization**:
   - Generates 3D scatter plots of the original and updated datasets.
4. **Logging**:
   - Provides detailed logs of operations and potential errors for debugging.
5. **Command-Line Interface**:
   - Flexible control over inputs, outputs, and parameters using CLI arguments.

---

## Running script
Before running these scripts, please ensure `pip` is installed on your system (see [here](https://pip.pypa.io/en/stable/installation/) for more information). The script can be run from the commandline:
```bash
pip install -r requirements.txt
python script.py 
```

or with a Docker image
```bash
docker built -t offset-points-generator .
docker run --rm -v "$(pwd)/outputs:/app/outputs" offset-points-generator
```
---

## Dependencies

Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `argparse`
- `matplotlib`
- `os`
  
## Workflow

### Input File Reading:
Reads the input dataset from a space-separated text file. Dataset is stored as a `pandas` DataFrame with columns: `label`,`x`,`y`,`z`.
For larger dimensionality data, an optional argument `point_cols` can be defined to label features accordingly.

### Offset Point Calculation:
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

### Output File Writing:
Saves the updated dataset, including new offset points, to the specified output file.

![Side-by-Side 3D Scatter Plots](plots/3d_scatter_plots.png "Side-by-Side 3D Scatter Plots")

Interactive plots:
[Interactive Plotly Chart Original Dataset](plots/original_3d_figure.html)
[Interactive Plotly Chart Updated Dataset](plots/updated_3d_figure.html)


## Error Handling

- `Missing Input File`: Logs an error and exits if the file is not found.
- `Invalid Parameters`: Validates all inputs and raises appropriate exceptions for invalid values.
- `Processing Errors`: Logs errors for individual points but continues processing the rest.

## Extensibility

- **Support for Higher Dimensions**: Modify `--point-cols` to handle additional dimensions.
- **Adjust Offset Behavior**: Customize the cutoff function to change how the offset is calculated.
- **Dynamic Labels**: Use custom labels for both input and output points.

## File Formatting 
### Input File Format

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
### Output File Format

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
## Command-Line Arguments

Run the script with the following command-line options:

### Optional Arguments
- `-i` or `--input`: Path to the input text file. (Default: `cdd.txt`)
- `-d` or `--offset-magnitude`: Magnitude of the offset. (Default: `2.0`)
- `-o` or `--output`: Path to the output text file. (Default: `out.txt`)
- `-a` or `--alpha`: Alpha value for the cutoff function. (Default: `0.5`)
- `--point-cols`: List of column names representing the coordinates. (Default: `['x', 'y', 'z']`)
- `-l` or `--label`: Label of the points to offset. (Default: `'B'`)
- `--offset-label`: Label to assign to the offset points. (Default: `'C'`)
### Example Command
```bash
python script.py -i input.txt -o output.txt -d 3.0 -a 0.5 --point-cols x y z -l B --offset-label C
```
## Example Workflow

### Input File (`cdd.txt`):
```
A 1.0 2.0 3.0
B 4.0 5.0 6.0
A 7.0 8.0 9.0
B 10.0 11.0 12.0
```
### Command:

`python script.py -i input.txt -o output.txt -d 3.0 -a 0.5 --point-cols x y z -l B --offset-label C`

### Output File (`out.txt`):
```
label    x     y     z
A        1.0   2.0   3.0
B        4.0   5.0   6.0
A        7.0   8.0   9.0
B        10.0  11.0  12.0
C        5.2   6.3   7.1
C        11.5  12.6  13.2
```

## **Function Documentation**

### 1. `cutoff_function`
Generates a polynomial decay function that transitions from a starting value to 0 over a defined number of steps.

#### **Parameters**
- `alpha` (float): Starting value of the function.
- `steps` (int): Number of steps for the transition.
- `n_degree` (int, optional): Degree of the polynomial (default: 3).

#### **Returns**
- `np.ndarray`: Array of decayed values.

#### **Raises**
- `ValueError`: If `steps <= 0` or `alpha < 0`.

---

### 2. `add_offset_points`
Applies offsets to points in a DataFrame based on PCA and outward vectors.

#### **Parameters**
- `df` (pd.DataFrame): Input DataFrame containing 3D points and labels.
- `offset_magnitude` (float): Magnitude of the offset.
- `point_cols` (list, optional): Column names for 3D coordinates (default: `['x', 'y', 'z']`).
- `initial_label` (str, optional): Label of points to offset (default: `'B'`).
- `offset_label` (str, optional): Label for new offset points (default: `'C'`).
- `alpha` (float, optional): Weight for the cutoff function (default: `0.5`).

#### **Returns**
- `pd.DataFrame`: Updated DataFrame with offset points added.

#### **Raises**
- `TypeError`: If `df` is not a DataFrame.
- `ValueError`: If `offset_magnitude <= 0`.
- `KeyError`: If required columns are missing in the DataFrame.

---

### 3. `plot_coordinates`
Creates 3D scatter plots of the original and updated datasets.

#### **Parameters**
- `df` (pd.DataFrame): Original dataset.
- `updated_df` (pd.DataFrame): Updated dataset.

#### **Notes**
- Saves the plot as `3d_scatter_plots.png`.

---

### 4. `main`
Entry point for the script. Parses command-line arguments and orchestrates the workflow.



### License

This script is open-source and can be freely modified or extended for your use 
