import unittest
import pandas as pd
import numpy as np
from script import cutoff_function, add_offset_points

class TestFunctions(unittest.TestCase):
    def test_cutoff_function_valid(self):
        """Test cutoff_function with valid inputs."""
        alpha = 1.0
        steps = 5
        n_degree = 3
        result = cutoff_function(alpha, steps, n_degree)
        self.assertEqual(len(result), steps)
        self.assertTrue(np.all(result <= alpha))  # Ensure all values are <= alpha
        self.assertTrue(np.all(result >= 0))  # Ensure no negative values

    def test_cutoff_function_invalid_steps(self):
        """Test cutoff_function raises error for invalid steps."""
        with self.assertRaises(ValueError):
            cutoff_function(1.0, -1, 3)

    def test_cutoff_function_invalid_alpha(self):
        """Test cutoff_function raises error for negative alpha."""
        with self.assertRaises(ValueError):
            cutoff_function(-1.0, 10, 3)

    def test_add_offset_points_valid(self):
        """Test add_offset_points with valid inputs."""
        data = {
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'z': [0, 1, 2],
            'label': ['B', 'B', 'A']
        }
        df = pd.DataFrame(data)
        result = add_offset_points(df, offset_magnitude=1.0)
        self.assertIn('label', result.columns)
        self.assertGreater(len(result), len(df))  # Ensure new points were added
        self.assertTrue(all(col in result.columns for col in ['x', 'y', 'z']))

    def test_add_offset_points_invalid_df(self):
        """Test add_offset_points raises error for invalid DataFrame."""
        with self.assertRaises(TypeError):
            add_offset_points([], 1.0)

    def test_add_offset_points_missing_columns(self):
        """Test add_offset_points raises error for missing columns."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with self.assertRaises(KeyError):
            add_offset_points(df, 1.0)

    def test_add_offset_points_invalid_offset(self):
        """Test add_offset_points raises error for invalid offset."""
        data = {
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'z': [0, 1, 2],
            'label': ['B', 'B', 'A']
        }
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError):
            add_offset_points(df, offset_magnitude=-1.0)

    def test_add_offset_points_empty_df(self):
        """Test add_offset_points handles empty DataFrame."""
        df = pd.DataFrame(columns=['x', 'y', 'z', 'label'])
        result = add_offset_points(df, offset_magnitude=1.0)
        self.assertEqual(len(result), 0)  # Ensure no points are added

    def test_cutoff_function_decay(self):
        """Test cutoff_function ensures a decaying output."""
        alpha = 1.0
        steps = 10
        n_degree = 3
        result = cutoff_function(alpha, steps, n_degree)
        self.assertTrue(np.all(np.diff(result) <= 0))  # Ensure the values decay

if __name__ == '__main__':
    unittest.main()

