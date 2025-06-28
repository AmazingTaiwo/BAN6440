##   BAN6440 Applied Machine Learning for Analytics
##   Module 4 Assignment - K-Means Python Application
##   Name: Taiwo Babalola
##   Learner ID: 162894
##   Submitted to: Rapheal Wanjiku

## Lunar Data Clustering using K-Means

## Overview
This project demonstrates the use of K-Means clustering on mock lunar observation data, simulating input from the [Kaguya Monoscopic Uncontrolled Observations](https://registry.opendata.aws/astrogeology/). It illustrates how unsupervised learning can be used to group unlabeled spatial data into meaningful clusters.

## Dataset
Due to access constraints, we simulate the dataset by creating mock `.img` files representing lunar surface data. Each file is processed to extract random numerical features (e.g., reflectance, elevation proxies), mimicking real feature extraction from satellite imagery.

## Features
- Simulates a realistic AWS-based spatial data ingestion pipeline
- Extracts mock features from `.img` files
- Applies data normalization using `StandardScaler`
- Clusters data using `KMeans` from `scikit-learn`
- Visualizes results using PCA-reduced 2D plots
- Includes unit tests for key functions

## Project Structure
  ├── kaguya_monoscopic_uncontrolled_observations/ # Simulated lunar files
  ├── AppName: ban6440_module_4_assignment_k_means.py # Main application code
  ├── AppName: module_4_assignment_k_means_unit_test.py # Unit tests
  ├── README.md


## How to Run
1. **Install dependencies** (recommended in a virtual environment):
   ```bash
   pip install numpy pandas matplotlib scikit-learn
2. Run the main script:
    python ban6440_module_4_assignment_k_means.py
3. Run unit tests:
   python -m unittest test_lunar_kmeans.py
4. Sample Output
    Clustering completes successfully and prints:
      [INFO] K-Means clustering completed with 3 clusters.
      A scatter plot is displayed showing the 2D PCA-transformed clusters.

## Requirements
   - Python 3.8+
   - Libraries: numpy, pandas, scikit-learn, matplotlib

## Author
   - Name: Taiwo Babalola
   - Course: BAN6440 – Applied Machine Learning for Analytics
