import os
import pandas as pd
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

def runparaview(main_folder, num_steps=6):

	# Prism element cell type for VTK
	vtk_prism_type = 13

	# Loop through each step folder
	for step in range(6, num_steps + 1):
		step_folder = os.path.join(main_folder, str(step))
		
		# Load nodal data
		nodal_data = pd.read_csv(f'{main_folder}/{step}/output_nodes.csv')

		# Load reactions data (residual forces)
		reactions_data = pd.read_csv(f'{main_folder}/{step}/output_reactions.csv')
		
		# Load connectivity data (only required node columns)
		connectivity_file = os.path.join(step_folder, 'output_elements.csv')
		connectivity_data = pd.read_csv(connectivity_file, usecols=['node1', 'node2', 'node3', 'node4', 'node5', 'node6'])
		connectivity_data2 = pd.read_csv(connectivity_file, usecols=['island_id'])#, 'mat_id', 'mat_id_features'])
		
		# Load residual force for free nodes
		residual_force_data = pd.read_csv(f'{main_folder}/{step}/residuals_free_step.csv')
		residual_force_data_2 = pd.read_csv(f'{main_folder}/{step}/residuals_free_step_2.csv')
		res_mask = pd.read_csv(f'{main_folder}/{step}/thresholding_res_mask.csv')
		res_mask_values = 10.0*(res_mask['res_mask'].values).astype(int)

		Highlights_mask = pd.read_csv(f'{main_folder}/{step}/Highlights.csv')
		
		Highlights_values = 10.0*(Highlights_mask['Highlights'].values.astype(int))

		# Extract residual free forces
		res_force_free = residual_force_data[['res_free_raw']].values.squeeze()
		res_force_free_2 = residual_force_data_2[['res_free_norm_2']].values.squeeze()
		# Create PyVista Mesh for current step
		points = nodal_data[['x', 'y', 'z']].values
		# Ensure points array is consistent
		if len(points) != len(nodal_data):
			print(f"Error: Mismatch in points array length for step {step}.")
			continue

		# Assign Dirichlet dofs (boundary conditions)
		bcs_nodes = nodal_data[['bcx', 'bcy', 'bcz']].values
		dirichlet_nodes_fix = (bcs_nodes != 0)	# True where the node is fixed
		
		# Flatten the boolean array
		dirichlet_nodes_fix = dirichlet_nodes_fix.flatten()	 # Flatten to a single vector (3*N,)
		
		# Total number of components (3 components per node: x, y, z)
		total_components = 3 * len(nodal_data)

		# Initialize res_x, res_y, res_z
		res_free_temp = np.zeros(total_components, dtype=np.float64)
		res_free_temp[~dirichlet_nodes_fix] = res_force_free
		
		res_free_temp_2 = np.zeros(total_components, dtype=np.float64)
		res_free_temp_2[~dirichlet_nodes_fix] = res_force_free_2
		
		res_free_mask = np.zeros(bcs_nodes.shape[0], dtype=np.float64)	  
		res_free_mask[(~dirichlet_nodes_fix).reshape(-1,3)[:,0]] = np.sum(Highlights_values.reshape(-1,3),axis=1)

		# Create masks for x, y, and z components
		mask_x = np.arange(0, total_components, 3)
		mask_y = np.arange(0, total_components, 3) + 1
		mask_z = np.arange(0, total_components, 3) + 2

		# Extract forces for each component
		res_x = res_free_temp[mask_x]
		res_y = res_free_temp[mask_y]
		res_z = res_free_temp[mask_z]
		res = np.column_stack((res_x, res_y, res_z))
		
		res_x = res_free_temp_2[mask_x]
		res_y = res_free_temp_2[mask_y]
		res_z = res_free_temp_2[mask_z]
		res_2 = np.column_stack((res_x, res_y, res_z))
		
		# Construct cell array: flatten connectivity and prepend the number of points per cell
		num_points_per_cell = 6	 # Prism elements have 6 nodes
		# breakpoint()
		connectivity_data = connectivity_data.values.astype(int) - 1
		cells = np.hstack([[num_points_per_cell] + list(connectivity_data[i]) for i in range(connectivity_data.shape[0])])
		# Create a cell types array (same type for each cell)
		cell_types = np.full(connectivity_data.shape[0], vtk_prism_type, dtype=np.uint8)

		# Check sizes
		print(f"Step {step}")
		print("Number of cells:", connectivity_data.shape[0])
		print("Number of points:", points.shape[0])
		if connectivity_data.shape[0] > 0:
			try:
				grid = pv.UnstructuredGrid(cells, cell_types, points)

				# Extract reaction forces
				reactions = reactions_data[['forces']].values.flatten()	 # Flatten to 1D array
				
				# Reshape reactions if necessary
				if len(reactions) == points.shape[0] * 6:
					reactions = reactions.reshape(-1, 6)  # Reshape into (n_nodes, 6)
					grid.point_data['ReactionForces'] = reactions
				
				# Add displacement vectors to the mesh
				displacements = nodal_data[['ux', 'uy', 'uz']].values
				grid.point_data['Displacement'] = displacements
				
				island_ids = connectivity_data2['island_id'].values
				grid.cell_data['Island_ids'] = island_ids
				
				# Add residual forces to the grid
				grid.point_data['ResidualForces'] = res
				grid.point_data['Res_2dmag'] = np.linalg.norm(res[:,0:2],axis=1)  
				grid.point_data['ResidualForces_2'] = res_2
				grid.point_data['Highlights'] = res_free_mask
				
				grid.save(f'{step_folder}/step_{step}_output.vtk')
			except Exception as e:
				print(f"Error creating UnstructuredGrid for step {step}: {e}")
		else:
			print(f"No cells found for step {step}. Skipping...")
if __name__=='__main__':
	runparaview('HW_NH_ISH_NH')