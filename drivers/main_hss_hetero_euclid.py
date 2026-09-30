import os
import sys
sys.path.insert(1, '../core/')
# os.environ["PYTHONBREAKPOINT"] = "0"

import numpy as np
import scipy.stats as stats
import copy
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle

from core_spike_slab import *
from unsupervised_hyperelasticity import *
from post_process import *
from helper import *
from preprocess_data_hetero import extractSystemOfEquations
from features_library import *

# from preprocess_data import extractSystemOfEquations
from config import *

# torch
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

# numpy
np.random.seed(0)

# matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# scipy
import scipy
from scipy import sparse

# pandas
import pandas as pd
from scipy.spatial import cKDTree

# others:
from contextlib import contextmanager
import shutil
from distutils.dir_util import copy_tree

# custom:
from data_definitions import *
from load_fem_data_wedge_2intpoints import *
from utilities import *
import subprocess
from growth_paraview_2_residualForces import *
from paraview_plot import*
import shutil
import time


from sklearn.cluster import KMeans
# from SolveF import process_fem_data_and_cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d, label
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# start_time = time.process_time()
start_time = time.time()


results = []
fem_dir = '../main_data/'

# Map noise level strings to numerical values
noise_map = {
	'none': 0,
	'low': 1e-3,
	'moderate': 2e-3,
	'high': 5e-3,
}


def collect_result(result):
	global results
	arr, ind = result
	results[ind]=arr
	
def main():
	rng = np.random.default_rng(seed)
	show_current_time()

	prefix = '../drivers'
	fig_prefix = 'Quasi-static '

	# Default values
	fem_material = None
	noiseLevel = 0.0

	# -------------------------------
	# Parse command-line arguments
	# -------------------------------
	if len(sys.argv) >= 3:
		fem_material = sys.argv[1]
		noise_level = sys.argv[2]
		noise_str = sys.argv[2].lower()

		if noise_str in noise_map:
			noiseLevel = noise_map[noise_str]
		else:
			raise ValueError(f"Unknown noise level '{noise_str}'. Choose from: {list(noise_map.keys())}")
	else:
		raise ValueError("Usage: python script.py <material_name> <noise_level>")

	# ----------------------------------------------------
	# Create KRR denoised copy if noiseLevel is not zero
	# ----------------------------------------------------
	if noiseLevel != 0.0:
		denoised_material = f"{fem_material}_denoised_{noiseLevel:.0e}".replace("+0", "")
		src_dir = os.path.join(fem_dir, fem_material)
		dst_dir = os.path.join(fem_dir, denoised_material)

		if not os.path.exists(dst_dir):
			shutil.copytree(src_dir, dst_dir)
			print(f"Created denoised folder: {dst_dir}")
		else:
			print(f"Denoised folder already exists: {dst_dir}")

		# Redirect to denoised folder for all processing
		fem_material = denoised_material


	# ----------------------------------------------------
	# Data preprocessing and system assembly
	# ----------------------------------------------------
	loadsteps = get_loadsteps(fem_material)
	feature_filter = get_feature_filter(fem_material)
	fem_dir_pp = fem_dir+'/'+fem_material
	filtered_data_set = []
	loadstep = 6

	X_elem2 = pd.read_csv(os.path.join(fem_dir, fem_material, str(6), 'output_elements.csv')) # Read as a DataFrame
	X_elem = X_elem2['node1'].to_numpy()	 # Extract 'node1' column and convert it to a NumPy array
	cluster_values = np.ones((X_elem.shape[0],), dtype=int)
	 
	extractSystemOfEquations(fem_dir_pp, [6], cluster_values.copy(), noiseLevel)

	raw_data = get_data(rng, fem_dir, prefix, fem_material, noise_level, loadstep, feature_filter)

	filtered_data = filter_raw_data(rng, raw_data, filter_value)

	filtered_data_set.append(filtered_data)

	data = process_raw_data(filtered_data_set, lambda_r)


	# Get number of clusters
	numfeatures = getNumberOfFeatures()
	numclusters = int(data.D.shape[1] / numfeatures)

	# Generate cluster indices to cover ranges like 0-3, 4-7, 8-11
	cluster_indices = []
	for i in range(numclusters):
		cluster_indices.extend([i * numfeatures, (i + 1) * numfeatures-1])
	cluster_indices_2 = (np.array(cluster_indices)).flatten().tolist()
	# Combine base indices with cluster-based indices
	selected_indices = cluster_indices_2

	# Ensure indices are within the bounds of data.D
	valid_indices = [idx for idx in selected_indices if idx < data.D.shape[1]]

	# Initialize theta_test as the initial guess
	tolerance = 1e-6  # Define a convergence tolerance for theta_ols
	loadstep = 6
	X_nodes = pd.read_csv(os.path.join(fem_dir, fem_material, str(6), 'output_nodes.csv')).to_numpy()

	theta_test_1 = np.linalg.lstsq(data.D[:, valid_indices], data.y, rcond=None)[0]
 

	A_free = pd.read_csv(f'{fem_dir_pp}/{str(loadstep)}/A1.csv', header=None).to_numpy()
	b_free = pd.read_csv(f'{fem_dir_pp}/{str(loadstep)}/b1.csv', header=None).to_numpy()

	A_fix = pd.read_csv(f'{fem_dir_pp}/{str(loadstep)}/A2.csv', header=None).to_numpy()
	b_fix = pd.read_csv(f'{fem_dir_pp}/{str(loadstep)}/b2.csv', header=None).to_numpy()


	# Filter A_free using valid indices
	A_free_valid = A_free[:, valid_indices]
	A_fix_valid = A_fix[:, valid_indices]

	# Calculate residual forces using theta_test
	res_free_raw = np.matmul(A_free_valid, np.transpose(theta_test_1)) - b_free.squeeze()
	res_fix_raw = np.matmul(A_fix_valid, np.transpose(theta_test_1)) - b_fix.squeeze()
	 
	# Save residual forces for this load step
	step_folder = f'{fem_dir_pp}/{str(loadstep)}'
	os.makedirs(step_folder, exist_ok=True)

	res_free_df = pd.DataFrame({'res_free_raw': res_free_raw})
	res_free_df.to_csv(f'{step_folder}/residuals_free_step.csv', index=False) 

	# Save fixed residuals for the current load step
	res_fix_df = pd.DataFrame({'res_fix_raw': res_fix_raw})
	res_fix_df.to_csv(f'{step_folder}/residuals_fix_step.csv', index=False)

	elements_file = os.path.join(step_folder, 'output_elements.csv')
	elements_data = pd.read_csv(elements_file)
	element_connectivity = elements_data[['node1', 'node2', 'node3', 'node4', 'node5', 'node6']].values

	# Update Element Data with Clusters

	growIslands(fem_dir_pp)

	output_elements_file = os.path.join(step_folder, 'output_elements.csv')
	output_elements_data = pd.read_csv(output_elements_file)

	print(f"Updated elements data with clusters saved to {output_elements_file}")
	 
	island_indices = output_elements_data['island_id'].values
	cluster_values = island_indices
	X_nodes = pd.read_csv(f'{fem_dir_pp}/{str(loadstep)}/output_nodes.csv')

	bcs_nodes = X_nodes[['bcx','bcy','bcz']].values
	nodes_list = np.arange(bcs_nodes.shape[0])
	nodes_list_internal = nodes_list[np.sum(bcs_nodes,axis=1)==0]

	flaggedNodesList = getFlaggedNodesList(total_num_nodes=X_nodes.shape[0],connectivity=torch.tensor(element_connectivity-1),cluster_ids_int_points=cluster_values,num_clusters=np.max(cluster_values))
	flaggedRowsList = (flaggedNodesList!=0)
	flaggedRows = flaggedRowsList[np.sum(bcs_nodes,axis=1)==0,:]
	temp = np.zeros((3*flaggedRows.shape[0],flaggedRows.shape[1]),dtype=bool)
	indices = np.arange(flaggedRows.shape[0])
	temp[3*indices,:] = flaggedRows
	temp[3*indices+1,:]= flaggedRows
	temp[3*indices+2,:]= flaggedRows
	flaggedRows = temp

	output_elements_data.to_csv(output_elements_file, index=False)

	extractSystemOfEquations(fem_dir_pp, [6], cluster_values.copy(), 0)

	numfeatures = getNumberOfFeatures()
	numclusters = int(np.max(cluster_values))

	# Get theta ols and res_force_2

	filtered_data_set = []

	raw_data = get_data(rng, fem_dir, prefix, fem_material, noise_level, loadstep, feature_filter)

	filtered_data = filter_raw_data(rng, raw_data,filter_value=[])
	filtered_data_set.append(filtered_data)

	data = process_raw_data(filtered_data_set, lambda_r)
	params = Params(N=data.D.shape[0],
				P=data.D.shape[1],
				a_v=a_v,
				b_v=b_v,
				a_p=a_p,
				b_p=b_p,
				a_sigma=a_sigma,
				b_sigma=b_sigma)
				
	theta_ols = np.linalg.lstsq(data.D, data.y, rcond=None)[0]

	A_free = pd.read_csv(f'{fem_dir_pp}/{str(loadstep)}/A1.csv', header=None).to_numpy()
	A_free_valid = A_free
	res_free_2 = A_free_valid@theta_ols
	res_free_2 = res_free_2.reshape(-1,3)
	res_free_2[:,2] = 0
	res_free_norm = np.linalg.norm(res_free_2,axis=1)
	res_free_norm = np.column_stack((res_free_norm,res_free_norm,res_free_norm)).flatten()
	
	res_fix_df_2 = pd.DataFrame({'res_free_norm_2': res_free_norm})
	res_fix_df_2.to_csv(f'{step_folder}/residuals_free_step_2.csv', index=False)

	filtered_data_set = []

	raw_data = get_data(rng, fem_dir, prefix, fem_material, noise_level, loadstep, feature_filter)

	filtered_data = filter_raw_data(rng, raw_data, filter_value=100, flaggedRows = flaggedRows,step_folder = step_folder)
	filtered_data_set.append(filtered_data)

	data = process_raw_data(filtered_data_set, lambda_r)
	params = Params(N=data.D.shape[0],
				P=data.D.shape[1],
				a_v=a_v,
				b_v=b_v,
				a_p=a_p,
				b_p=b_p,
				a_sigma=a_sigma,
				b_sigma=b_sigma)

	cluster_indices = []
	for i in range(numclusters):
		cluster_indices.extend([i * numfeatures, (i + 1) * numfeatures-1])
	cluster_indices_2 = (np.array(cluster_indices)).flatten().tolist()

	# Combine base indices with cluster-based indices
	selected_indices = cluster_indices_2

	#Compute theta_ols using least squares
	valid_indices = [idx for idx in selected_indices if idx < data.D.shape[1]]

	theta_ols_2 = np.linalg.lstsq(data.D, data.y, rcond=None)[0]
	
	theta_ols_2 = theta_ols_2.reshape(-1, numfeatures)
	runparaview(fem_dir_pp)

		
	global results
	# Run Bayesian Spike-and-Slab MCMC Chains
	results = [Chain(params, chain_length, burn)]*parallel_chains
	pool = mp.Pool(parallel_chains)
	for chain_iter in range(parallel_chains):
		pool.apply_async(run_spike_slab,args=(rng, data, params, chain_length, burn, chain_iter, True), callback=collect_result)
		 
	pool.close()
	pool.join()
	pool.terminate()

	# Combine all chains into one
	combined_chain = results[0]
	for chain_iter in range(1, parallel_chains):
		combined_chain.combine_chain(results[chain_iter])
	
	with open(step_folder+'/chain.pkl', 'wb') as f:
		pickle.dump(combined_chain, f)
	
	 
	# Final theta should be of size (total number of features across all clusters)
	theta = np.zeros(data.D.shape[1])

	# Use Bayesian average (or filtered average) from the combined chain
	theta = np.mean(combined_chain.theta, axis=0)
	theta = theta.reshape(-1, numfeatures)

	# Optional: Standard deviation for uncertainty
	std_theta = np.zeros_like(theta)
	std_theta = np.std(combined_chain.theta, axis=0).reshape(theta.shape)
	
	print(f"Mean Theta: {theta}")
	print(f"std: {std_theta}")
	end_time = time.time()
	cpu_time = end_time - start_time
	print("CPU time:", end_time - start_time, "seconds")
	 
	# Build list of rows
	rows = []
	for i in range(theta.shape[0]):
		row = {f'theta_{j}': theta[i, j] for j in range(theta.shape[1])}
		row.update({f'std_{j}': std_theta[i, j] for j in range(std_theta.shape[1])})
		row['cpu_time'] = cpu_time
		rows.append(row)

	# Create DataFrame and save
	df = pd.DataFrame(rows)
	# Save to CSV
	output_path = fem_dir_pp+'/'+str(loadstep)+'/'+'theta_stats.csv'
	df.to_csv(output_path, index=False)

	print(f"Saved theta, std_theta, and CPU time to {output_path}")
	
	# Convert to DataFrame
	df_ols = pd.DataFrame(theta_ols_2, columns=[f'theta_ols_{j}' for j in range(theta.shape[1])])

	# Define path and save
	ols_output_path = fem_dir_pp + '/' + str(loadstep) + '/' + 'theta_ols.csv'
	df_ols.to_csv(ols_output_path, index=False)
	show_current_time()

if __name__=='__main__':
	main()