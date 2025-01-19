
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import logging
import argparse
import matplotlib.pyplot as plt 
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cutoff_function(alpha, steps, n_degree=3):
    """
    Generates a cutoff function that smoothly transitions from `alpha` to `0` in `steps` using
    a polynomial of degree `n_degree`.

    Parameters:
    ----------
    alpha : float
        The starting value of the function.
    steps : int
        The number of steps (or samples) over which the transition occurs.
    n_degree : int, optional (default=3)
        The degree of the polynomial used for the cutoff (e.g., 1 for linear, 2 for quadratic).

    Returns:
    -------
    np.ndarray
        A 1D array of `steps` values representing the cutoff function.

    Raises:
    ------
    ValueError
        If `steps` is less than or equal to 0, or if `alpha` is negative.
    """
    if steps <= 0:
        raise ValueError("Parameter 'steps' must be greater than 0.")
    if alpha < 0:
        raise ValueError("Parameter 'alpha' must be non-negative.")
    
    # Create a linear space from 0 to 1
    x = np.linspace(0, 1, steps)
    # Compute the cutoff values using the polynomial decay formula
    return alpha * np.exp(-n_degree * x)


def add_offset_points(df, offset_magnitude, point_cols=['x', 'y', 'z'], initial_label='B', offset_label='C', alpha=0.5):
    """
    Adds offset points to a DataFrame based on points with a specific initial label, creating new points 
    at a fixed distance and in a direction pointing "outward" from the point cloud.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame containing the original point cloud data. Each row represents a point, 
        with columns for coordinates (e.g., x, y, z) and labels.

    offset_magnitude : float
        The magnitude (distance) of the offset to apply to the new points.

    point_cols : list of str, optional (default=['x', 'y', 'z'])
        The names of the columns in `df` representing the coordinates of the points in 3D space. Can be extended for more dimensions.

    initial_label : str, optional (default='B')
        The label of the points in `df` to which the offset operation will be applied.

    offset_label : str, optional (default='C')
        The label assigned to the newly generated offset points.

    alpha : float, optional (default=0.5)
        The starting weight for the cutoff function, determining the emphasis of outward directionality.

    Returns:
    -------
    pandas.DataFrame
        The updated DataFrame containing the original points and the new offset points.

    Raises:
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    ValueError
        If `offset_magnitude` is not positive.
    KeyError
        If `point_cols` or the 'label' column are missing in the DataFrame.

    Notes:
    -----
    - PCA is used to determine the principal axes of the point cloud.
    - The direction of the offset combines an outward vector and weighted principal axes.
    """
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Parameter 'df' must be a pandas DataFrame.")
    if offset_magnitude <= 0:
        raise ValueError("Parameter 'offset_magnitude' must be greater than 0.")
    if not all(col in df.columns for col in point_cols):
        raise KeyError(f"Columns {point_cols} are not all present in the DataFrame.")
    if 'label' not in df.columns:
        raise KeyError("The DataFrame must contain a 'label' column.")

    logging.info(f"Starting add_offset_points with offset_magnitude={offset_magnitude}, alpha={alpha}")

    # Drop missing values 
    df = df.dropna()

    # Filter points with the specified initial label
    labeled_points = df[df['label'] == initial_label]
    logging.info(f"Found {len(labeled_points)} points with label '{initial_label}'")

    for idx, point in labeled_points.iterrows():
        try:
            # Compute the centroid of the point cloud
            centroid = df[point_cols].mean()

            # Perform PCA to find the principal axes of the point cloud
            pca = PCA(n_components=len(point_cols))
            pca.fit(df[point_cols])
            principal_axes = pca.components_

            # Compute the outward vector from the centroid to the current point
            outward_vector = point[point_cols] - centroid
            outward_vector /= np.linalg.norm(outward_vector)  # Normalize to unit vector

            # Generate weights for the principal axes
            weights = cutoff_function(alpha=alpha, steps=len(point_cols), n_degree=3)
            # Compute the offset direction as a weighted sum of the outward vector and principal axes
            direction = outward_vector + np.dot(weights, principal_axes)
            direction /= np.linalg.norm(direction)  # Normalize again

            # Create a new offset point
            offset_point = point.copy()
            offset_point[point_cols] += direction * offset_magnitude
            offset_point['label'] = offset_label
            # Append the offset point to the DataFrame
            df = pd.concat([df, offset_point.to_frame().T], ignore_index=True)

        except Exception as e:
            logging.error(f"Error processing point at index {idx}: {e}")
            continue

    logging.info(f"Completed adding offset points. Total points in DataFrame: {len(df)}")
    return df

def plot_coordinates(df, updated_df):
    """ Plots the original and updated datasets in 3D scatter plots.

    Args:
        df (pd.DataFrame): Original dataset.
        updated_df (pd.DataFrame): Updated dataset with offset points.
    """
    # Define a colormap for unique labels
    unique_labels = updated_df['label'].unique()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create the dictionary mapping labels to colors
    color_map = {label: colors[i] for i,label in enumerate(unique_labels)}
    
    # Map the colors based on the 'label' column
    df['color'] = df['label'].map(color_map)
    updated_df['color'] = updated_df['label'].map(color_map)
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(12, 6))

    # First subplot for Dataset 1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(df['x'], df['y'], df['z'], c=df['color'])
    ax1.set_title("Original Dataset")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Second subplot for Dataset 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(updated_df['x'], updated_df['y'], updated_df['z'], c=updated_df['color'])
    ax2.set_title("Updated Dataset")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label, 
                                    markerfacecolor=color_map[label], markersize=10) for label in color_map.keys()])

    # Save the plot as a PNG file
    output_path_colored = "outputs/plots/3d_scatter_plots.png"
    os.makedirs(os.path.dirname(output_path_colored), exist_ok=True)
    plt.savefig(output_path_colored)
    logging.info(f"Saved 3D scatter plot to {output_path_colored}")

def main():
    """
    Main function for running the script from the command line.

    Command-Line Arguments:
    -----------------------
    -i, --input : str (required)
        Path to the input text file containing the dataset.

    -o, --output : str (default='out.txt')
        Path to the output text file where results will be saved.

    -d, --offset-magnitude : float (required)
        The magnitude of the offset to apply.

    -a, --alpha : float (default=0.5)
        The alpha value for the cutoff function.

    --point-cols : list of str (default=['x', 'y', 'z'])
        List of column names representing the 3D coordinates.

    -l, --label : str (default='B')
        Label of points to offset.

    --offset-label : str (default='C')
        Label to assign to the offset points.

    Raises:
    ------
    IOError
        If the input file cannot be read or the output file cannot be written.
    """
    parser = argparse.ArgumentParser(
        description="Add offset points to a 3D dataset. The script calculates outward offsets for points "
                    "with a specific label and adds them to the dataset."
    )
    parser.add_argument("-i", "--input", default='cdd.txt', help="Path to the input text file.")
    parser.add_argument("-o", "--output", default="out.txt", help="Path to the output text file.")
    parser.add_argument("-d", "--offset-magnitude", type=float, default=2.0, help="Magnitude of the offset.")
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="Alpha value for the cutoff function.")
    parser.add_argument("--point-cols", nargs="+", default=["x", "y", "z"], help="List of coordinate column names.")
    parser.add_argument("-l", "--label", default="B", help="Label of points to offset.")
    parser.add_argument("--offset-label", default="C", help="Label for the offset points.")
    parser.add_argument("-p","--plot", default=True, help="Plot the original and updated datasets.")
    
    args = parser.parse_args()

    try:
        # Define the column names as 'label' + components of point_cols
        column_names = ['label'] + args.point_cols
        # Read the input file and assign column names
        df = pd.read_csv(args.input, sep=" ", header=None, names=column_names)
        logging.info(f"Loaded input file {args.input} with columns: {column_names}")
        
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        raise IOError("Failed to read input file.")

    try:
        # Add offset points to the DataFrame
        updated_df = add_offset_points(
            df,
            offset_magnitude=args.offset_magnitude,
            point_cols=args.point_cols,
            initial_label=args.label,
            offset_label=args.offset_label,
            alpha=args.alpha,
        )
    except Exception as e:
        logging.error(f"Error processing offset points: {e}")
        raise RuntimeError("Error in offset point calculation.")

    try:
        # Save the updated DataFrame to the specified output file
        os.makedirs(os.path.dirname(f'output