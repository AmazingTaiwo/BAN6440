#   BAN6440 Applied Machine Learning for Analytics
#   Module 4 Assignment - K-Means Python Application
#   Name: Taiwo Babalola
#   Learner ID: 162894
#   Submitted to: Rapheal Wanjiku
#   AppName: module_4_assignment_k_means_unit_test.py
#   Author Taiwo Babalola


import unittest
import os
import numpy as np
import pandas as pd
from ban6440_module_4_assignment_k_means import extract_mock_features, run_kmeans_clustering

class TestLunarKMeans(unittest.TestCase):

    def setUp(self):
        """Set up a mock directory with dummy image files for testing."""
        self.test_dir = "test_mock_lunar"
        os.makedirs(self.test_dir, exist_ok=True)
        for i in range(3):
            with open(os.path.join(self.test_dir, f"test_{i+1}.img"), "w") as f:
                f.write("test content")
        self.df = extract_mock_features(self.test_dir)

    def tearDown(self):
        """Clean up test files."""
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_feature_extraction_shape(self):
        """Test that correct number of features are extracted."""
        self.assertEqual(self.df.shape[0], 3)
        self.assertEqual(self.df.shape[1], 6)  # 5 features + 1 file_name

    def test_kmeans_cluster_count(self):
        """Test K-Means returns correct number of cluster labels."""
        labels, model, scaled = run_kmeans_clustering(self.df, n_clusters=2)
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_error_on_too_few_samples(self):
        """Test that clustering fails with too few samples."""
        small_df = self.df.iloc[:1]
        with self.assertRaises(ValueError):
            run_kmeans_clustering(small_df, n_clusters=3)

    def test_error_on_missing_directory(self):
        """Test that missing directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            extract_mock_features("non_existent_dir")

# Redirect output to a log file
with open("unit_test_results.txt", "w") as f:
    runner = unittest.TextTestRunner(stream=f, verbosity=2)
    suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')  # adjust if your test file is named differently
    result = runner.run(suite)

if __name__ == "__main__":
    # Run unit tests and save output to a log file
    with open("unit_test_results.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLunarKMeans)
        runner.run(suite)

    # Also print results to console
    with open("unit_test_results.txt", "r") as f:
        print(f.read())
