# Particle Filter SLAM

### Directory Structure

ECE276A_PR2_Local/...<br />
├─── code/...<br />
│ ├─── first_lidar_scan.py/...<br />
│ ├─── slam.py/...<br />
│ ├─── mapping.py/...<br />
│ ├─── prediction.py/...<br />
│ ├─── update.py/...<br />
│ ├─── resampling.py/...<br />
│ ├─── pr2_utils.py/...<br />
│ ├─── fog_delta_yaw.npy/...<br />
├─── param/...<br />
├─── saved_map/...<br />
├─── saved_wall/...<br />
├─── saved_texture_map/...<br />
├─── sensor_data/...<br />
├─── stereo_images/...<br />

### Main Files under code directory
* **first_lidar_map.py**: use to plot first lidar scan onto an occupancy grid map, uncomment plotting in mapping.py to use
* **slam.py**: main code to run particle filter slam
* **mapping.py**: given lidar scan, finds all occupied and free cells with bresenham2D, updates grid map
* **prediction.py**: calculates new vehicle pose for each particle given linear velocity and change in yaw
* **update.py**: updates particle weights based on which lidar and pose have highest correlation
* **resampling.py**: creates new set of particles
* **texture_map.py**: projects rgb image to map using disparity
* **pr2_utils.py**: methods for read_data_from_csv, mapCorrelation, bresenham2D, compute_stereo
* **fog_delta.npy**: fog data synced to encoder timestamps, sync code commented out in slam.py

### Other Folders
* **param**: sensor property values and transforms
* **saved_map**: occupancy grid map saved here every 10k iterations
* **saved_wall**: occupancy grid map walls saved here every 10k iterations
* **saved_texture_map**: occupancy grid map with texture saved here every 10k iterations
* **sensor_data**: contains fog, encoder, lidar data
* **stereo_images.py**: contains stereo camera image data

### Notes
* **stereo_images** and **sensor_data* directories are not included as they are too large

### How to Run

Download modules in requirements.txt<br />
Run slam.py <br />
There is a live updated map shown as the car drives <br />
Outputs map and wall images saved every 10k iterations in saved_map and saved_wall directories

