import os
import torch
import numpy as np
import pandas as pd
from utilities import *
from features_library import *
from config import *
from data_definitions import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import loguniform
# import pyvista as pv


def loadFemData(path, AD=True, noiseLevel=0, noiseType='displacement', denoisedDisplacements=None):
	"""
	Load finite element data and add noise (optional), followed by KRR denoising of displacement.

	_Input Arguments_
	- `path` - path to finite element data
	- `AD` - specify if automatic differention is needed
	- `noiseLevel` - noise level
	- `noiseType` - a string or list of strings: 'displacement','strain','acceleration','velocity'
	- `denoisedDisplacements` - pass denoised displacement data if already available

	_Output Arguments_
	- `dataset` - finite element dataset
	"""
	print('\n-----------------------------------------------------')
	print('Loading data: ', path)
	if path[-1] == '/':
		path = path[:-1]

	numNodesPerElement = 6

	# Load nodal data
	df = pd.read_csv(path + '/output_nodes.csv', dtype=np.float64)
	numNodes = df.shape[0]
	x_nodes = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float64)
	u_nodes = torch.tensor(df[['ux', 'uy', 'uz']].values, dtype=torch.float64)
	n_int_points = 2

	if ('vx' in df.columns) and ('vy' in df.columns):
		v_nodes = torch.tensor(df[['vx', 'vy', 'vz']].values, dtype=torch.float64)
	else:
		v_nodes = torch.zeros_like(u_nodes)

	if ('ax' in df.columns) and ('ay' in df.columns):
		a_nodes = torch.tensor(df[['ax', 'ay', 'az']].values, dtype=torch.float64)
	else:
		a_nodes = torch.zeros_like(u_nodes)

	if denoisedDisplacements is not None:
		u_nodes = denoisedDisplacements

	bcs_nodes = torch.tensor(df[['bcx', 'bcy', 'bcz']].round().astype(int).values)
	dirichlet_nodes = (bcs_nodes != 0)

	if isinstance(noiseType, str):
		noiseType = [noiseType]

	if not any(nt in ['displacement', 'strain', 'velocity', 'acceleration'] for nt in noiseType):
		raise ValueError('Incorrect noiseType argument!')
	
	# -------------Spatially correlated noise----------------
	# corr_length = 1.0          # Spatial correlation length (in same unit as x,y,z)
	# jitter = 1e-6              # For numerical stability

	# diff = x_nodes.unsqueeze(1) - x_nodes.unsqueeze(0)
	# dist_matrix = torch.norm(diff, dim=2)  # Euclidean distance matrix [numNodes, numNodes]

	# # Construct exponential covariance kernel and Cholesky factor
	# C = torch.exp(-dist_matrix / corr_length)  # Exponential spatial kernel
	# C = C / C.max()
	# L = torch.linalg.cholesky(C + jitter * torch.eye(numNodes, dtype=torch.float64))  # Cholesky factorization

	#  Uncorrelated noise to displacements
	noise_u_nodes = noiseLevel * torch.randn_like(u_nodes)

	if 'displacement' in noiseType and noiseLevel>1e-10:

		# Uncorrelated noise to displacements
		u_nodes += noise_u_nodes

		# Spatially correlated noise >> Apply correlation independently for each displacement component (x,y,z)
		# for i in range(3):
		# 	u_nodes[:, i] += L @ noise_u_nodes[:, i]

		print('Applying noise to displacements:', noiseLevel)
		
		df_noised = df.copy()
		df_noised[['ux', 'uy', 'uz']] = u_nodes
		
		material_folder = os.path.join(path)
		os.makedirs(material_folder, exist_ok=True)
		
		# Save to material subfolder
		output_path_2 = os.path.join(material_folder, 'output_nodes_with_noised_disp.csv')
		df_noised.to_csv(output_path_2, index=False)
		print('Saved noised node data to:', output_path_2)

		# Save to the same output_nodes.csv file (OVERWRITE)
		# output_path = os.path.join(material_folder, 'output_nodes.csv')
		# df_noised.to_csv(output_path, index=False)
		# print('Saved noised node data to:', output_path)

		#-------------------------------
		# KRR Denoising of displacements
		#-------------------------------
		# Denoise using Kernel Ridge Regression (auto-optimized)
		print('Performing KRR denoising...')
		X_np = x_nodes.numpy()
		u_noisy_np = u_nodes.numpy()
		u_true_np = df[['ux', 'uy', 'uz']].values  # true values before noise
		u_denoised_np = np.zeros_like(u_noisy_np)
		for i in range(3):	# For ux, uy, uz
			model = KernelRidge(alpha= 1e-03, kernel='rbf', gamma=5.0)
			# model = KernelRidge(alpha= 2e-01, kernel='laplacian', gamma=15.0)
			model.fit(X_np, u_noisy_np[:, i])
			u_denoised_np[:, i] = model.predict(X_np)

		# Replace tensor
		u_nodes = torch.tensor(u_denoised_np, dtype=torch.float64)
		print('KRR denoising complete.')

		# Save denoised data
		# Overwrite displacement columns in the original DataFrame
		df[['ux', 'uy', 'uz']] = u_denoised_np

		# Save to the same output_nodes.csv file (overwrite)
		output_path = os.path.join(material_folder, 'output_nodes.csv')
		df.to_csv(output_path, index=False)

		print('Overwrote original output_nodes.csv with denoised displacement data at:', output_path)
	 
	# Add noise to accelerations
	noise_a_nodes = noiseLevel * torch.randn_like(a_nodes)
	noise_a_nodes[dirichlet_nodes] = 0.
	if 'acceleration' in noiseType:
		a_nodes += noise_a_nodes
		print('Applying noise to accelerations:', noiseLevel)

	# Add noise to velocities
	noise_v_nodes = noiseLevel * torch.randn_like(v_nodes)
	noise_v_nodes[dirichlet_nodes] = 0.
	if 'velocity' in noiseType:
		v_nodes += noise_v_nodes
		print('Applying noise to velocities:', noiseLevel)

	# Load reaction forces
	
	numReactions = torch.max(bcs_nodes).item()
	df = pd.read_csv(path + '/output_reactions.csv', dtype=np.float64)
	reactions = []
	for i in range(numReactions):
		reactions.append(Reaction(bcs_nodes == (i + 1), df['forces'][i]))

	# Load element data
	df = pd.read_csv(path + '/output_elements.csv', dtype=np.float64)
	connectivity = np.zeros((df.shape[0], numNodesPerElement), dtype=np.int32)
	for i in range(numNodesPerElement):
		connectivity[:, i] = df['node' + str(i + 1)].to_numpy() - 1
	 
	# Extract X, Y, Z coordinates for 6-node elements (triangular prism)
	Xcordelms = x_nodes[connectivity, 0]
	Ycordelms = x_nodes[connectivity, 1]
	Zcordelms = x_nodes[connectivity, 2]
	
	# Build the matrix for node positions
	Pmatrix = np.column_stack((Xcordelms[:, 0], Ycordelms[:, 0], Zcordelms[:, 0]))
	for k in range(1, connectivity.shape[1]):
		Pmatrix = np.column_stack((Pmatrix, Xcordelms[:, k], Ycordelms[:, k], Zcordelms[:, k]))

	# Define node positions for the triangular prism
	p1 = torch.tensor(Pmatrix[:, 0:3])	# Node 1 (lower triangle)
	p2 = torch.tensor(Pmatrix[:, 3:6])	# Node 2 (lower triangle)
	p3 = torch.tensor(Pmatrix[:, 6:9])	# Node 3 (lower triangle)
	p4 = torch.tensor(Pmatrix[:, 9:12])	 # Node 4 (upper triangle)
	p5 = torch.tensor(Pmatrix[:, 12:15]) # Node 5 (upper triangle)
	p6 = torch.tensor(Pmatrix[:, 15:18]) # Node 6 (upper triangle)
	

	
	# nodes_stack = torch.column_stack((p1, p2, p3, p4, p5, p6))
	
	# Compute the centroid
	X0 = torch.mean(torch.column_stack((p1[:,0], p2[:,0], p3[:,0], p4[:,0], p5[:,0], p6[:,0])), dim=1)
	X1 = torch.mean(torch.column_stack((p1[:,1], p2[:,1], p3[:,1], p4[:,1], p5[:,1], p6[:,1])), dim=1)
	X2 = torch.mean(torch.column_stack((p1[:,2], p2[:,2], p3[:,2], p4[:,2], p5[:,2], p6[:,2])), dim=1)
	
	X0.requires_grad=True
	X1.requires_grad=True
	# X2.requires_grad=True
   
	# X= torch.column_stack((X0,X1,X2)) # These are the coordinates of the centroid.
	# X = torch.tensor(X, dtype=torch.float64, requires_grad=True)
	#X.requires_grad = True
	
	# height1 = np.abs(p4[:, 2] - p1[:, 2])
	# height2 = np.abs(p5[:, 2] - p2[:, 2])
	# height3 = np.abs(p6[:, 2] - p3[:, 2])
	
	# Average height for robustness
	topz = torch.mean(torch.column_stack((p4[:,2],p5[:,2],p6[:,2])),dim=1)
	botz = torch.mean(torch.column_stack((p1[:,2],p2[:,2],p3[:,2])),dim=1)
	
	X2_intp1 = X2 - (topz-botz)/6
	X2_intp2 = X2 + (topz-botz)/6
	
	X2_intp1.requires_grad=True
	X2_intp2.requires_grad=True
	
	# Create ppVol and reshape as needed
	ppVol = 0.5 * torch.norm(torch.cross(p1 - p2, p3 - p2),dim=1)*torch.abs(topz - botz)
	ppVol = torch.row_stack((ppVol,ppVol)).flatten()/2
	 
	#ppVol = ppVol.reshape(-1, 1)

	# Create test indices (1-based to 0-based)
	# testindices = torch.linspace(1, ppVol.shape[0], ppVol.shape[0], dtype=torch.int32) - 1

	# Check condition where ppVol <= 0
	# testcond = (ppVol <= 0).squeeze()	 # This will return a boolean tensor, no need for torch.sum here

	# Filter ppVol based on the test condition
	# ppVol = ppVol[~testcond]

	# Update numElements based on filtered ppVol
	numElements = ppVol.shape[0]

	# Apply testcond to connectivity and elems
	# connectivity = connectivity[~testcond,:]
	# elems = elems[~testcond,:]
	
	# Define the shape functions for the lower triangle (z-component included)
	def compute_N1(X0,X1,X2,p1,p2,p3):
		# X = torch.column_stack((X0,X1,X2))
		zerocol = torch.zeros(X0.shape[0],1)
		X = torch.column_stack((X0,X1,p2[:,2]))
		 
		temp1 = (torch.norm(torch.cross(X- p2, p3 - p2),dim=1)/torch.norm(torch.cross(p1 - p2, p3 - p2),dim=1))
		temp2 = (topz - X2)/(topz-botz)
		# Shape function for N1 at an arbitrary point X (with cross product approach)
		return	temp1*temp2
	
	# Similarly for N2, N3, N4, N5, N6
	def compute_N2(X0,X1,X2,p1,p2,p3):
		zerocol = torch.zeros(X0.shape[0],1)
		X = torch.column_stack((X0,X1,p1[:,2]))
		return (torch.norm(torch.cross(X - p1, p3 - p1),dim=1) /
				torch.norm(torch.cross(p2 - p1, p3 - p1),dim=1)) * (topz - X2)/(topz-botz)
	
	def compute_N3(X0,X1,X2,p1,p2,p3):
		zerocol = torch.zeros(X0.shape[0],1)
		X = torch.column_stack((X0,X1,p1[:,2]))
		return (torch.norm(torch.cross(X - p1, p2 - p1),dim=1) /
				torch.norm(torch.cross(p3 - p1, p2 - p1),dim=1)) * (topz - X2)/(topz-botz)
	
	# Upper face for nodes 4, 5, 6
	def compute_N4(X0,X1,X2,p4,p5,p6):
		zerocol = torch.zeros(X0.shape[0],1)
		X = torch.column_stack((X0,X1,p5[:,2]))
		return (torch.norm(torch.cross(X - p5, p6 - p5),dim=1) /
				torch.norm(torch.cross(p4 - p5, p6 - p5),dim=1)) * (X2-botz)/(topz-botz)
	
	def compute_N5(X0,X1,X2,p4,p5,p6):
		zerocol = torch.zeros(X0.shape[0],1)
		X = torch.column_stack((X0,X1,p4[:,2]))
		return (torch.norm(torch.cross(X - p4, p6 - p4),dim=1) /
				torch.norm(torch.cross(p5 - p4, p6 - p4),dim=1)) * (X2-botz)/(topz-botz)
		
	def compute_N6(X0,X1,X2,p4,p5,p6):
		zerocol = torch.zeros(X0.shape[0],1)
		X = torch.column_stack((X0,X1,p4[:,2]))
		return (torch.norm(torch.cross(X - p4, p5 - p4),dim=1) /
				torch.norm(torch.cross(p6 - p4, p5 - p4),dim=1)) * (X2-botz)/(topz-botz)
	
		# Compute shape functions
	N1_i1 = compute_N1(X0,X1,X2_intp1,p1,p2,p3)
	N2_i1 = compute_N2(X0,X1,X2_intp1,p1,p2,p3)
	N3_i1 = compute_N3(X0,X1,X2_intp1,p1,p2,p3)
	N4_i1 = compute_N4(X0,X1,X2_intp1,p4,p5,p6)
	N5_i1 = compute_N5(X0,X1,X2_intp1,p4,p5,p6)
	N6_i1 = compute_N6(X0,X1,X2_intp1,p4,p5,p6)
	
	N1_i2 = compute_N1(X0,X1,X2_intp2,p1,p2,p3)
	N2_i2 = compute_N2(X0,X1,X2_intp2,p1,p2,p3)
	N3_i2 = compute_N3(X0,X1,X2_intp2,p1,p2,p3)
	N4_i2 = compute_N4(X0,X1,X2_intp2,p4,p5,p6)
	N5_i2 = compute_N5(X0,X1,X2_intp2,p4,p5,p6)
	N6_i2 = compute_N6(X0,X1,X2_intp2,p4,p5,p6)
	
	N1 = torch.concat((N1_i1,N1_i2))
	N2 = torch.concat((N2_i1,N2_i2))
	N3 = torch.concat((N3_i1,N3_i2))
	N4 = torch.concat((N4_i1,N4_i2))
	N5 = torch.concat((N5_i1,N5_i2))
	N6 = torch.concat((N6_i1,N6_i2))

	 
	# # Compute gradients of shape functions with respect to X
	# N1.backward(retain_graph=True)
	# gradN1 = X.grad.clone()	 # Clone to keep the gradient for later
	# X.grad.zero_()
	# N2.backward(retain_graph=True)
	# gradN2 = X.grad.clone()
	# X.grad.zero_()
	# N3.backward(retain_graph=True)
	# gradN3 = X.grad.clone()
	# X.grad.zero_()
	# N4.backward(retain_graph=True)
	# gradN4 = X.grad.clone()
	# X.grad.zero_()
	# N5.backward(retain_graph=True)
	# gradN5 = X.grad.clone()
	# X.grad.zero_()
	# N6.backward(retain_graph=True)
	# gradN6 = X.grad.clone()
	# X.grad.zero_()
	 
	


	# Initialize gradient tensors
	# gradN1 = torch.zeros((X0.shape[0], 3), dtype=X0.dtype)
	# gradN2 = torch.zeros((X0.shape[0], 3), dtype=X0.dtype)
	# gradN3 = torch.zeros((X0.shape[0], 3), dtype=X0.dtype)
	# gradN4 = torch.zeros((X0.shape[0], 3), dtype=X0.dtype)
	# gradN5 = torch.zeros((X0.shape[0], 3), dtype=X0.dtype)
	# gradN6 = torch.zeros((X0.shape[0], 3), dtype=X0.dtype)
	
	def sdot(e,f):
		temp2 = torch.zeros(e.shape[0],1)
		 
		for i in range(e.shape[1]):
			temp2 += (e[:,i]*f[:,i]).reshape(-1,1)
		return temp2.reshape(-1,1)
	
	area = torch.norm(torch.cross(p1-p2,p3-p2),dim=1)
	# area2 = torch.norm(torch.cross(p5-p4,p6-p4),dim=1)
	
	#gradN1[:,0] = ((1 / torch.sqrt((X-p2)*specialdot(p3-p2,p3-p2)-(p3-p2)*specialdot(X-p2,p3-p2)))/area)*(topz-X[:,2])/(topz-botz)
	def product(g,h):
		temp = g*(sdot(h,h))-h*(sdot(g,h))
		 
		return temp
		
	#X = [X0,X1,X2]
	 
	# def compute_gradN1(X0,X1,X2,p1,p2,p3):
		# zerocol = torch.zeros(X0.shape[0],1)
		# X = torch.column_stack((X0,X1,zerocol))
		#  
		# temp1 = (product(X - p2, p3 - p2)/((torch.sqrt(sdot(torch.cross(X - p2, p3 - p2), torch.cross(X - p2, p3 - p2))))*area.reshape(-1,1)))
		# grad = temp1*(topz-X2).reshape(-1,1)/(topz-botz).reshape(-1,1)
		# N1_temp = (torch.norm(torch.cross(X - p2, p3 - p2),dim=1)/torch.norm(torch.cross(p1 - p2, p3 - p2),dim=1))
		#  
		# grad[:,2] = (-N1_temp.reshape(-1,1)/(topz - botz).reshape(-1,1)).squeeze()
		# return grad
		
	# gradN1_theory = compute_gradN1(X0,X1,X2,p1,p2,p3)
	
	 
	 
	def compute_gradients(N,X0,X1,X2):
		X = [X0,X1,X2]
		grad_tensor = torch.zeros((X0.shape[0], 3), dtype=X0.dtype)	 # Initialize gradient tensor
		 
		for i in range(3):	# Iterate over X0, X1, X2
			grad_output = torch.ones(X0.shape[0],1) # Create the correct-sized grad_outputs
			grad = torch.autograd.grad(N.reshape(-1,1),X[i], grad_outputs=grad_output, create_graph=True, retain_graph=True, allow_unused=True)[0]
			
			# Handle None case
			if grad is not None:
				grad_tensor[:, i] = grad
			else:
				grad_tensor[:, i] = torch.zeros_like(X[:, i])	 # Assign zero if no gradient exists
		return grad_tensor

	# Compute gradients for each shape function with respect to X0, X1, X2
	gradN1_i1 = compute_gradients(N1_i1,X0,X1,X2_intp1)
	gradN2_i1 = compute_gradients(N2_i1,X0,X1,X2_intp1)
	gradN3_i1 = compute_gradients(N3_i1,X0,X1,X2_intp1)
	gradN4_i1 = compute_gradients(N4_i1,X0,X1,X2_intp1)
	gradN5_i1 = compute_gradients(N5_i1,X0,X1,X2_intp1)
	gradN6_i1 = compute_gradients(N6_i1,X0,X1,X2_intp1)
	
	gradN1_i2 = compute_gradients(N1_i2,X0,X1,X2_intp2)
	gradN2_i2 = compute_gradients(N2_i2,X0,X1,X2_intp2)
	gradN3_i2 = compute_gradients(N3_i2,X0,X1,X2_intp2)
	gradN4_i2 = compute_gradients(N4_i2,X0,X1,X2_intp2)
	gradN5_i2 = compute_gradients(N5_i2,X0,X1,X2_intp2)
	gradN6_i2 = compute_gradients(N6_i2,X0,X1,X2_intp2)
	 
	
	gradN1 = torch.row_stack((gradN1_i1,gradN1_i2))
	gradN2 = torch.row_stack((gradN2_i1,gradN2_i2))
	gradN3 = torch.row_stack((gradN3_i1,gradN3_i2))
	gradN4 = torch.row_stack((gradN4_i1,gradN4_i2))
	gradN5 = torch.row_stack((gradN5_i1,gradN5_i2))
	gradN6 = torch.row_stack((gradN6_i1,gradN6_i2))
	
	# Compute deformation gradient tensor based on the displacements
	connectivity = np.row_stack((connectivity,connectivity))
	u1 = u_nodes[connectivity, 0]
	u2 = u_nodes[connectivity, 1]
	u3 = u_nodes[connectivity, 2]

	# Compute displacement vectors for nodes
	uN1 = torch.column_stack((u1[:, 0], u2[:, 0], u3[:, 0]))
	uN2 = torch.column_stack((u1[:, 1], u2[:, 1], u3[:, 1]))
	uN3 = torch.column_stack((u1[:, 2], u2[:, 2], u3[:, 2]))
	uN4 = torch.column_stack((u1[:, 3], u2[:, 3], u3[:, 3]))
	uN5 = torch.column_stack((u1[:, 4], u2[:, 4], u3[:, 4]))
	uN6 = torch.column_stack((u1[:, 5], u2[:, 5], u3[:, 5]))
	 
	F11 = 1+uN1[:,0]*gradN1[:,0]+uN2[:,0]*gradN2[:,0]+uN3[:,0]*gradN3[:,0]+uN4[:,0]*gradN4[:,0]+uN5[:,0]*gradN5[:,0]+uN6[:,0]*gradN6[:,0]
	F12 = uN1[:,0]*gradN1[:,1]+uN2[:,0]*gradN2[:,1]+uN3[:,0]*gradN3[:,1]+uN4[:,0]*gradN4[:,1]+uN5[:,0]*gradN5[:,1]+uN6[:,0]*gradN6[:,1]
	F13 = uN1[:,0]*gradN1[:,2]+uN2[:,0]*gradN2[:,2]+uN3[:,0]*gradN3[:,2]+uN4[:,0]*gradN4[:,2]+uN5[:,0]*gradN5[:,2]+uN6[:,0]*gradN6[:,2]
	F21 = uN1[:,1]*gradN1[:,0]+uN2[:,1]*gradN2[:,0]+uN3[:,1]*gradN3[:,0]+uN4[:,1]*gradN4[:,0]+uN5[:,1]*gradN5[:,0]+uN6[:,1]*gradN6[:,0]
	F22 = 1+uN1[:,1]*gradN1[:,1]+uN2[:,1]*gradN2[:,1]+uN3[:,1]*gradN3[:,1]+uN4[:,1]*gradN4[:,1]+uN5[:,1]*gradN5[:,1]+uN6[:,1]*gradN6[:,1]
	F23 = uN1[:,1]*gradN1[:,2]+uN2[:,1]*gradN2[:,2]+uN3[:,1]*gradN3[:,2]+uN4[:,1]*gradN4[:,2]+uN5[:,1]*gradN5[:,2]+uN6[:,1]*gradN6[:,2]
	F31 = uN1[:,2]*gradN1[:,0]+uN2[:,2]*gradN2[:,0]+uN3[:,2]*gradN3[:,0]+uN4[:,2]*gradN4[:,0]+uN5[:,2]*gradN5[:,0]+uN6[:,2]*gradN6[:,0]
	F32 = uN1[:,2]*gradN1[:,1]+uN2[:,2]*gradN2[:,1]+uN3[:,2]*gradN3[:,1]+uN4[:,2]*gradN4[:,1]+uN5[:,2]*gradN5[:,1]+uN6[:,2]*gradN6[:,1]
	F33 = 1+uN1[:,2]*gradN1[:,2]+uN2[:,2]*gradN2[:,2]+uN3[:,2]*gradN3[:,2]+uN4[:,2]*gradN4[:,2]+uN5[:,2]*gradN5[:,2]+uN6[:,2]*gradN6[:,2]
	
# F11 = 1+uN1[0]*gradN1[:,0]+uN2[0]*gradN2[:,0]+uN3[0]*gradN3[:,0]+uN4[0]*gradN4[:,0]
# F12 = uN1[0]*gradN1[:,1]+uN2[0]*gradN2[:,1]+uN3[0]*gradN3[:,1]+uN4[0]*gradN4[:,1]
# F13 = uN1[0]*gradN1[:,2]+uN2[0]*gradN2[:,2]+uN3[0]*gradN3[:,2]+uN4[0]*gradN4[:,2]
# F21 = uN1[1]*gradN1[:,0]+uN2[1]*gradN2[:,0]+uN3[1]*gradN3[:,0]+uN4[1]*gradN4[:,0]
# F22 = 1+uN1[1]*gradN1[:,1]+uN2[1]*gradN2[:,1]+uN3[1]*gradN3[:,1]+uN4[1]*gradN4[:,1]
# F23 = uN1[1]*gradN1[:,2]+uN2[1]*gradN2[:,2]+uN3[1]*gradN3[:,2]+uN4[1]*gradN4[:,2]
# F31 = uN1[2]*gradN1[:,0]+uN2[2]*gradN2[:,0]+uN3[2]*gradN3[:,0]+uN4[2]*gradN4[:,0]
# F32 = uN1[2]*gradN1[:,1]+uN2[2]*gradN2[:,1]+uN3[2]*gradN3[:,1]+uN4[2]*gradN4[:,1]
# F33 = 1+uN1[2]*gradN1[:,2]+uN2[2]*gradN2[:,2]+uN3[2]*gradN3[:,2]+uN4[2]*gradN4[:,2]
	#
	# np.savetxt('output_exp_defgrads.csv',np.column_stack((F11,F22,F33,F12,F13,F23,F21,F31,F32)),header='F11,F22,F33,F12,F13,F23,F21,F31,F32', fmt='%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e,%20.20e', comments='')

	# Read CSV and reshape data for connectivity
	df = pd.read_csv(path + '/output_elements.csv', dtype=np.float64)

	# Fill connectivity based on filtered data
	connectivity = []
	for i in range(numNodesPerElement):
		temp = df['node' + str(i + 1)].round().astype(int) - 1
		temp = np.row_stack((temp,temp)).flatten()
		connectivity.append(torch.tensor(temp.tolist()))

	# Set up quadrature point weights and gradients
	qpWeights = ppVol.squeeze()
	 
	gradNa = [gradN1, gradN2, gradN3, gradN4, gradN5, gradN6]
	Na = [N1,N2,N3,N4,N5,N6]
	#
	# #element-wise rearrangement of displacements
	# u = []
	# for i in range(numNodesPerElement):
	#	u.append(u_nodes[connectivity[i],:])
	#

	F = torch.column_stack((F11,F12,F21,F22,F13,F23,F31,F32,F33)).detach()
	# Fall = np.column_stack((F,F,F,F))
	# Fall = np.reshape(Fall,(-1,9))
	# mask = ~np.isnan(Fall)
	# mask = np.prod(mask,axis=1,dtype=np.bool_)
	# elems2 = np.reshape(elems,(-1,1))
	# Fnodesall = np.zeros((x_nodes.shape[0],9))
	#  
	# np.add.at(Fnodesall,elems2[mask].squeeze(),Fall[mask.squeeze(),:])

	# ons = np.ones(elems2.shape)
	# denoms = np.zeros(x_nodes.shape[0])
	# # denoms=denoms.squeeze()
	# # np.add.at(denoms,elems[mask].squeeze(),ons[mask,:].squeeze())
	# np.add.at(denoms,elems2[mask].squeeze(),ons[mask].squeeze())
	# # denoms[denoms==0]=1
	# denoms=np.reshape(denoms,(-1,1))
	# Fnodesall = Fnodesall/denoms
	# indchange = [0,1,4,2,3,5,6,7,8]
	# Fnodesall = Fnodesall[:,indchange]
	# detF = Fnodesall[:,0]*(Fnodesall[:,4]*Fnodesall[:,8]-Fnodesall[:,7]*Fnodesall[:,5]) - Fnodesall[:,1]*(Fnodesall[:,3]*Fnodesall[:,8]-Fnodesall[:,6]*Fnodesall[:,5]) + Fnodesall[:,2]*(Fnodesall[:,3]*Fnodesall[:,7]-Fnodesall[:,4]*Fnodesall[:,6])

	# nelems = 4*np.ones((elems.shape[0],),dtype=np.int32)
	# elems2 = np.column_stack((nelems,elems))
	# elems2 = elems2.flatten()
	# celltypes = 10*np.ones(elems.shape[0],dtype=np.int32)
	#  
	# grid = pv.UnstructuredGrid(elems2, celltypes, x_nodes.numpy())
	# grid.point_data['displacement'] = u_nodes.numpy()
	# grid.point_data['DG'] = Fnodesall
	# grid.point_data['detF'] = detF
	# grid.save(path+'/EUCLIDvtkcheck.vtk')
	# F = torch.tensor(F)
	#computing deformation gradient at quadrature points
	# dim=3
	# voigtMap = [[0,1,4],[2,3,5],[6,7,8]]
	#
	# F=torch.zeros(numElements,9)
	# for a in range(numNodesPerElement):
	#	for i in range(dim):
	#		for j in range(dim):
	#			F[:,voigtMap[i][j]] += u[a][:,i] * gradNa[a][:,j]
	#
	# F[:,0] += 1.0
	# F[:,3] += 1.0
	# F[:,8] += 1.0

	if('strain' in noiseType):
		F += noiseLevel * torch.randn_like(F)
		print('Applying noise to strains:',noiseLevel)
	 
	#computing detF
	J = computeJacobian(F)

	#computing Cauchy-Green strain: C = F^T F
	C = computeCauchyGreenStrain(F)

	#computing strain invariants
	I1, I2, I3 = computeStrainInvariants(C)
	# If they are tensors, convert them to numpy arrays for saving
	# I1_np = I1.cpu().detach().numpy()
	# I2_np = I2.cpu().detach().numpy()
	# I3_np = I3.cpu().detach().numpy()
	# loadstep=6
	# Define the directory and filename for saving
	# # save_dir = fem_dir + '/' + str(loadstep)
	# # os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

	# Save the invariants as CSV files using np.savetxt
	# np.savetxt(fem_dir+'/'+str(loadstep)+'/I1.csv', I1_np, delimiter=',')
	# np.savetxt(fem_dir+'/'+str(loadstep)+'/I2.csv', I2_np, delimiter=',')
	# np.savetxt(fem_dir+'/'+str(loadstep)+'/I3.csv', I3_np, delimiter=',')
	
	K1 = (I1) * torch.pow(I3,-1/3) - 3.0
	# Some standard terms
	# K1 = I1 * torch.pow(I3,-1/3) - 3.0
	K2 = (I2) * torch.pow(I3,-2/3) - 3.0
	J = torch.sqrt(I3)
	J = (J-1.0)**2.0
	K1_np = K1.detach().cpu().numpy() 
	K2_np = K2.detach().cpu().numpy() 
	K3_np = J.detach().cpu().numpy()
	# Create an ID column
	num_rows = K1_np.shape[0]
	ID = np.arange(1, num_rows + 1)	 # Row numbers starting from 1

	# Specify output directory
	loadstep = 6
	output_dir = path
	#os.makedirs(output_dir, exist_ok=True)

	# Save strain invariants and IDs to a CSV file
	np.savetxt(
		f'{output_dir}/invariants.csv',
		np.column_stack((ID, K1_np, K2_np, K3_np)),	 # Combine ID and invariants
		header='ID,K1,K2,K3',
		fmt='%d,%20.20e,%20.20e,%20.20e',  # Format for ID and invariants
		comments=''
	)

	print("Strain invariants saved to 'invariants.csv'")
		

	# a_mult is defined in config
	Ia = C @ a_mult
	Ib = C @ b_mult

	#activate gradients
	I1.requires_grad = True
	I2.requires_grad = True
	I3.requires_grad = True
	Ia.requires_grad = True
	Ib.requires_grad = True

	#computing invariant derivaties
	dI1dF = computeStrainInvariantDerivatives(F,1)
	dI2dF = computeStrainInvariantDerivatives(F,2)
	dI3dF = computeStrainInvariantDerivatives(F,3)
	dIadF = computeStrainInvariantDerivatives(F,4)
	dIbdF = computeStrainInvariantDerivatives(F,5)

	# Ijunk.detach()
	#computing extended set of nonlinear features
	featureSet = FeatureSet()
	featureSet.features = computeFeatures_torch(I1, I2, I3, Ia, Ib) #don't detach, need it for autograd
	 
	if(AD==True):
		def differentiateFeaturesWithInvariants(features,I):
			"""
			Compute derivatives of the features with respect to the invariants of the Cauchy-Green strain tensor using automatic differentiation.

			_Input Arguments_

			- `features` - features

			- `I` - invariant

			_Output Arguments_

			- `d_feature_dI` - derivative

			---

			"""
			d_feature_dI = torch.zeros(features.shape[0],features.shape[1])
			for i in range(features.shape[1]):
				 
				temp = torch.autograd.grad(features[:,i:(i+1)],I,torch.ones(I.shape[0],1),create_graph=True,allow_unused=True)[0]
				 # temp = torch.autograd.grad(features[:,i:(i+1)],I,torch.ones(I.shape[0],1),create_graph=True,allow_unused=True)[0]
				if(type(temp)!=type(None)):
					d_feature_dI[:,i:(i+1)] = temp
			return d_feature_dI
		 
		featureSet.d_features_dI3 = differentiateFeaturesWithInvariants(featureSet.features,I3)
		featureSet.d_features_dI1 = differentiateFeaturesWithInvariants(featureSet.features,I1)
		featureSet.d_features_dI2 = differentiateFeaturesWithInvariants(featureSet.features,I2)
		featureSet.d_features_dIa = differentiateFeaturesWithInvariants(featureSet.features,Ia)
		featureSet.d_features_dIb = differentiateFeaturesWithInvariants(featureSet.features,Ib)


		#detach features now:
		featureSet.features = featureSet.features.detach()
		featureSet.d_features_dI1 = featureSet.d_features_dI1.detach()
		featureSet.d_features_dI2 = featureSet.d_features_dI2.detach()
		featureSet.d_features_dI3 = featureSet.d_features_dI3.detach()
		featureSet.d_features_dIa = featureSet.d_features_dIa.detach()
		featureSet.d_features_dIb = featureSet.d_features_dIb.detach()

	#density
	if os.path.exists(path+'/output_density.csv'):
		df = pd.read_csv(path+'/output_density.csv',dtype=np.float64)
		density = df.density[0]
	else:
		print('\nWARNING: density file not found\n')
		density = 1.4e-9
	lumped_mass = torch.zeros(numNodes, 3)
	
	
	# for a1 in range(numNodesPerElement):
		# lumped_mass[:,0].index_add_(0,connectivity[a1],qpWeights*density*1./6.0)
		# lumped_mass[:,1].index_add_(0,connectivity[a1],qpWeights*density*1./6.0)
		# lumped_mass[:,2].index_add_(0,connectivity[a1],qpWeights*density*1./6.0)
	
	# for e in range(numElements):
		# for a1 in range(numNodesPerElement):
			# lumped_mass[connectivity[a1][e]][0] += density * qpWeights[e]
			# lumped_mass[connectivity[a1][e]][1] += density * qpWeights[e]
			# lumped_mass[connectivity[a1][e]][2] += density * qpWeights[e]

	# lumped mass_acceleration | size: numNodes x 2
	lumped_mass_acceleration = torch.mul(lumped_mass,a_nodes)
	 
	dataset = FemDataset(path,
		x_nodes, u_nodes, v_nodes, a_nodes, dirichlet_nodes,
		reactions,
		connectivity, gradNa, Na, qpWeights,
		F, J, C,
		I1, I2, I3, Ia, Ib,
		dI1dF, dI2dF, dI3dF, dIadF, dIbdF,
		featureSet,
		density,
		lumped_mass_acceleration)

	print('-----------------------------------------------------\n')

	return dataset
