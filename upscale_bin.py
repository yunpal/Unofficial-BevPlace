import os
import shutil
import numpy as np
import open3d as o3d
from tqdm import tqdm 

def load_bin_file(filename):
    points = np.fromfile(filename, dtype=np.float64).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def scale_points(pcd, scale_factor=20):

    points = np.asarray(pcd.points) * scale_factor
    scaled_pcd = o3d.geometry.PointCloud()
    scaled_pcd.points = o3d.utility.Vector3dVector(points)
    return scaled_pcd

def save_bin_file(pcd, filename):
    np_points = np.asarray(pcd.points)
    np_points.astype(np.float64).tofile(filename)

def process_directory(source_directory, target_directory, scale_factor=20):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for item in tqdm(os.listdir(source_directory)):
        source_path = os.path.join(source_directory, item)
        target_path = os.path.join(target_directory, item)

        if os.path.isdir(source_path):
            process_directory(source_path, target_path, scale_factor)
        elif item.endswith('.csv'):
            shutil.copy2(source_path, target_path)
        elif item.endswith('.bin'):
            pcd = load_bin_file(source_path)
            scaled_pcd = scale_points(pcd, scale_factor)
            target_bin_file = target_path.replace('.bin', '.bin')
            save_bin_file(scaled_pcd, target_bin_file)

def transfer_and_process_directories(benchmark_path, target_path, scale_factor=20):
    process_directory(benchmark_path, target_path, scale_factor)

benchmark_datasets_path = '/data/soomoklee/data/benchmark_datasets'
target_directory_path = '/data/soomoklee/data/benchmark_datasets_upscaled'

# Start the process
transfer_and_process_directories(benchmark_datasets_path, target_directory_path)
