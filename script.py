
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import logging
import argparse
import matplotlib.pyplot as plt 
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_offset_direction(df,point,n_neighbors=5,point_cols=['x', 'y', 'z']):
    """
        Finds the offset direction for a given point in a DataFrame using general outward vector and PCA.
        
        Parameters:
        df : pandas.DataFrame
            The DataFrame containing the point cloud data.
        point : pandas.Series
            The point for which the offset direction is to be calculated.
        n_neighbors : int, optional (default=5)
            The number of nearest neighbors to consider for PCA.
        point_cols : list of str, optional (default=['x', 'y', 'z'])
            The names of the columns in `df` representing the coordinates of the points in 3D space. Can be extended to more dimensions.
            
        Returns:
        np.ndarray
            The offset direction as a unit vector.
    """
    # Format point cloud and query point for consistency
    point_cloud = df[point_cols]
    query_point = pd.DataFrame([point[point_cols].values],columns=point_cols)
    
    # Find the outward vector from the point cloud center to the point
    outward_vector = query_point.values[0] - point_cloud.mean()
    unit_outward_vector = outward_vector / np.linalg.norm(outward_vector)  # Normalize to unit vector
    
    # Find local geometry of the point using KNN and PCA
    knn = NearestNeighbors(n_neighbors=n_neighbors,algorithm='auto')
    knn.fit(point_cloud)
    distances, indices = knn.kneighbors(query_point)
    neighbor_list = point_cloud.iloc[indices[0]]

    pca = PCA(n_components=len(point_cols))
    pca.fit(neighbor_list)
    principal_axes = pca.components_
    principal_values = pca.explained_variance_ratio_
    
    # Offset direction accounting for local geometry
    direction = outward_vector + np.dot(principal_values, principal_axes)
    unit_direction = direction / np.linalg.norm(direction)  # Normalize 
    
    return unit_direction

def add_offset_points(df, offset_magnitude, point_cols=['x', 'y', 'z'], initial_label='B', offset_label='C', n_neighbors=5):
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
        
    n_neighbors : int, optional (default=5)
        The number of nearest neighbors to consider for PCA.

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

    logging.info(f"Starting add_offset_points with offset_magnitude={offset_magnitude}, n_neighbors={n_neighbors}")    
    
    # Clean df and filter points with the specified initial label
    df = df.dropna()
    labeled_points = df[df['label'] == initial_label]
    logging.info(f"Found {len(labeled_points)} points with label '{initial_label}'")

    for idx, point in labeled_points.iterrows():
        try:
            # Compute the centroid of the point cloud
            direction = find_offset_direction(df, point, point_cols=point_cols, n_neighbors=n_neighbors)
            
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
    parser.add_argument("--point-cols", nargs="+", default=["x", "y", "z"], help="List of coordinate column names.")
    parser.add_argument("-l", "--label", default="B", help="Label of points to offset.")
    parser.add_argument("--offset-label", default="C", help="Label for the offset points.")
    parser.add_argument("-p","--plot", default=True, help="Plot the original and updated datasets.")
    parser.add_argument("-n", "--neighbors", type=int, default=5, help="Number of neighbors for PCA.")
    
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
            offset_label=args.offset_label
            )
        logging.info("Updated database with offset points.")
    except Exception as e:
        logging.error(f"Error processing offset points: {e}")
        raise RuntimeError("Error in offset point calculation.")

    try:
        # Save the updated DataFrame to the specified output file
        os.makedirs(os.path.dirname(f'outputs/{args.output}'), exist_ok=True)
        updated_df.to_csv(f'outputs/{args.output}', sep=" ", index=False, header=False)
        logging.info(f"Saved output to: outputs/{args.output}")
    except Exception as e:
        logging.error(f"Error writing output file: {e}")
        raise IOError("Failed to write output file.")
    
    if args.plot and len(args.point_cols) == 3:
        try:
            # Plot the original and updated datasets
            plot_coordinates(df, updated_df)
        except Exception as e:
            logging.error(f"Error plotting datasets: {e}")
            raise RuntimeError("Failed to plot datasets.")

    # Output the updated DataFrame to the console
    print(updated_df[['label', *args.point_cols]])
    
if __name__ == "__main__":
    main()
