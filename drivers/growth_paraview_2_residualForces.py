import pandas as pd
import numpy as np
# import meshio
import os
import torch
from itertools import combinations
import sys
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import scipy.signal as ss

# folder_path = sys.argv[1]

def getFlaggedNodes(total_num_nodes,connectivity,cluster_ids_int_points,num_clusters):
    # num_clusters = np.max(cluster_ids_int_points)
    num_clusters = int(num_clusters)
    itering = combinations(list(range(1,num_clusters+1)),2)
    # list_global_nodes_flag = []
    global_nodes_flag = torch.zeros((total_num_nodes,),dtype=torch.int32)
    for iter in itering:
        nodes_flag_1 = torch.zeros((total_num_nodes,),dtype=torch.int32)
        nodes_flag_2 = torch.zeros((total_num_nodes,),dtype=torch.int32)
         
        int_point_flag_1 = torch.tensor(iter[0]==cluster_ids_int_points,dtype=torch.int32)
        int_point_flag_2 = torch.tensor(iter[1]==cluster_ids_int_points,dtype=torch.int32)
         
        for a in range(connectivity.shape[1]):
            nodes_flag_1.index_add_(0,connectivity[:,a],int_point_flag_1.squeeze())
            nodes_flag_2.index_add_(0,connectivity[:,a],int_point_flag_2.squeeze())
        global_nodes_flag += nodes_flag_1*nodes_flag_2
     
    return global_nodes_flag

def getFlaggedNodesList(total_num_nodes,connectivity,cluster_ids_int_points,num_clusters):
    # num_clusters = np.max(cluster_ids_int_points)
    num_clusters = int(num_clusters)
    itering = combinations(list(range(1,num_clusters+1)),2)
    list_global_nodes_flag = []
    global_nodes_flag = torch.zeros((total_num_nodes,),dtype=torch.int32)
    for iter in itering:
        nodes_flag_1 = torch.zeros((total_num_nodes,),dtype=torch.int32)
        nodes_flag_2 = torch.zeros((total_num_nodes,),dtype=torch.int32)
         
        int_point_flag_1 = torch.tensor(iter[0]==cluster_ids_int_points,dtype=torch.int32)
        int_point_flag_2 = torch.tensor(iter[1]==cluster_ids_int_points,dtype=torch.int32)
         
        for a in range(connectivity.shape[1]):
            nodes_flag_1.index_add_(0,connectivity[:,a],int_point_flag_1.squeeze())
            nodes_flag_2.index_add_(0,connectivity[:,a],int_point_flag_2.squeeze())
        global_nodes_flag += nodes_flag_1*nodes_flag_2
        list_global_nodes_flag.append(nodes_flag_1*nodes_flag_2)
     
    return np.column_stack(list_global_nodes_flag)
    
    
def deleteLayer(total_num_nodes, FlaggedNodes, connectivity, degree=3):
    count = 0
    nodes_flag = torch.zeros((total_num_nodes,))
     
    onesy = torch.ones((connectivity.shape[0],1))
    int_point_flag = torch.zeros((connectivity.shape[0],),dtype=torch.bool)
    while(count<degree):
        for a in range(connectivity.shape[1]):
            connectivity_col = connectivity[:,a].reshape((-1,1))
            FlaggedNodes = FlaggedNodes.reshape((1,-1))
            temp = torch.tensor(connectivity_col-FlaggedNodes,dtype=torch.bool)
            int_point_flag = int_point_flag + (torch.prod(temp,dim=1,dtype=torch.int32) == 0).squeeze()
        for a in range(connectivity.shape[1]):
            connectivity_col = connectivity[:,a].reshape((-1,1))
             
            nodes_flag.index_add_(0,connectivity_col[int_point_flag].squeeze(),onesy[int_point_flag].squeeze())
         
        temp = torch.arange(total_num_nodes)
         
        FlaggedNodes_new = temp[nodes_flag!=0].reshape(-1,1)
         
        tempyy = (FlaggedNodes_new - FlaggedNodes)!=0
        edge_mask = (torch.prod(tempyy,dim=1))
         
        FlaggedNodes_edge = FlaggedNodes_new[edge_mask]
         
        FlaggedNodes = FlaggedNodes_new
        count=count+1
        
     
    return FlaggedNodes,int_point_flag,FlaggedNodes_edge


def growIslands(folder_path,loadstep=6):# Load CSVs
    elements_df = pd.read_csv(f'{folder_path}/{loadstep}/output_elements.csv')  # node1, ..., node6, mat_id
    nodes_df = pd.read_csv(f'{folder_path}/{loadstep}/output_nodes.csv')        # id, x, y, z, bcx, bcy, bcz
    # Convert to NumPy arrays
    
    elements_np = elements_df.values.astype('int') # shape: (num_elements, 7)
    nodes_np = nodes_df[['id', 'x', 'y', 'z', 'bcx', 'bcy', 'bcz']].values
    elements_np[:,0:6] = elements_np[:,0:6] - 1
    element_global_indices = np.arange(elements_np.shape[0])
    res_forces = np.zeros((nodes_df.shape[0],))
    machineeps = 1e-9
    free_nodes_mask = np.abs((np.sum(nodes_np[:,4:],axis=1)))<machineeps
    res_force_data = pd.read_csv(f'{folder_path}/{loadstep}/residuals_free_step.csv')
    res_force_data = res_force_data['res_free_raw'].values
    res_force_data = res_force_data.reshape(-1,3)
    res_force_data = np.linalg.norm(res_force_data[:,0:2],axis=1)
    range_ = (np.max(res_force_data)-np.min(res_force_data))
    bin_width = 1e-3
    histo = plt.hist(res_force_data,bins = int(np.ceil(range_/bin_width)))[0]
    actual_bin_width = range_/histo.shape[0]
    peaks = ss.find_peaks(histo,threshold=4)[0]
    
    selection_width = 2
     
    for iter in range(1,selection_width):
        peaks1 = peaks+iter
        peaks2 = peaks-iter
        peaks3 = np.union1d(peaks1,peaks2)
        peaks3 = peaks3[peaks3>=0]
        peaks = np.union1d(peaks,peaks3)
    
    mask = np.zeros(res_force_data.shape,dtype=bool)
    for iter in range(peaks.shape[0]):
         
        mask = mask + (res_force_data>=actual_bin_width*peaks[iter]) #& (res_force_data<=actual_bin_width*(peaks[iter]+1.))
    
    res_mask = pd.DataFrame({'res_mask': mask})
    res_mask.to_csv(f'{folder_path}/{loadstep}/thresholding_res_mask.csv', index=False)

    # Standard deviation
    std_dev = np.std(res_force_data)
    mean_resforce = np.mean(res_force_data)
    print(f"MEAN: sigma_fres_hom = {mean_resforce}")
    print(f"STD_DEV: sigma_fres_hom = {std_dev}")

    # Assign to free nodes
    res_forces[free_nodes_mask] = res_force_data
    
    element_global_islands = np.zeros((elements_np.shape[0],))
    total_num_nodes = nodes_np.shape[0]
    temp = torch.arange(nodes_np.shape[0])

    # lambda_res is tunable scalar (refer Eq 28 in Hetero EUCLID paper)
    lambda_res = 1.5
    threshold = lambda_res * std_dev
    FlaggedNodes_edge = (temp[res_forces>threshold].reshape(-1,1)+1).squeeze().detach().numpy()
     
    int_point_flag = torch.ones((elements_np.shape[0],),dtype=torch.bool)
     
    elements_np = elements_np[int_point_flag,0:6]
    # elements_np = elements_np[:,0:6]

     
    node_max = np.max(elements_np)
    nodes_large_list = torch.zeros((node_max+1,),dtype=torch.int)
    # nodes_large_list_2 = torch.arange(node_max+1,dtype=torch.int)
    for a in range(elements_np.shape[1]):
        nodes_large_list.index_add_(0,torch.tensor(elements_np[:,a]),torch.ones_like(torch.tensor(elements_np[:,a])))
    boolmask = (nodes_large_list>0)
    num_nodes = torch.sum(boolmask,dtype = torch.int)
    new_nodes = torch.arange(num_nodes,dtype=torch.int)
    nodes_large_list[boolmask] = new_nodes
    temp = torch.arange(nodes_large_list.shape[0])
    elements_np = nodes_large_list[elements_np[:,0:6]].detach().numpy()+1
     
    nodes_np = nodes_np[temp[boolmask],:]
    FlaggedNodes_edge = nodes_large_list[FlaggedNodes_edge-1].detach().numpy()+1
    nodes_np[:,0] = np.arange(nodes_np.shape[0])+1
    flag_for_nodes = np.zeros((nodes_np.shape[0],))
    flag_for_nodes[FlaggedNodes_edge-1]=100
     
    # Step 1: Build node-to-element map using reverse lookup
    num_elements, max_nodes = elements_np.shape
    node_ids = nodes_np[:, 0].astype(int)
    coords = nodes_np[:, 1:4]


    # Build node_to_elems: map from node_id to list of connected elements
    flat_nodes = elements_np.flatten()
    element_indices = np.repeat(np.arange(num_elements), max_nodes)
    valid_mask = ~np.isnan(flat_nodes)
     
    flat_nodes = flat_nodes[valid_mask].astype(int)
    element_indices = element_indices[valid_mask]

    # For each node, get which elements it's in
    from collections import defaultdict
    node_to_elems = defaultdict(set)
    for node, eid in zip(flat_nodes, element_indices):
        node_to_elems[node].add(eid)

    # Step 2: Identify boundary nodes
    boundary_nodes = FlaggedNodes_edge

    # Step 3: Identify boundary elements
    elem_ids = np.arange(num_elements)
    elements_int = elements_np.astype('int')
    boundary_element_flag = np.zeros(elem_ids.shape)
    # boundary_element_flag[boundary_counts > 0] = 100
    for k in range(elements_int.shape[1]):
        boundary_element_flag += np.sum((elements_int[:,k].reshape(-1,1)-boundary_nodes.reshape(1,-1))==0,axis=1)
     
    boundary_counts = boundary_element_flag.copy()
    boundary_elements = set(elem_ids[boundary_element_flag > 0])

    end_flag = 1
    no_of_islands = 0

    elem_indices = np.arange(elem_ids.shape[0])
    elements_covered = 0
    element_to_island = {}
    total_domain = set()
    total_growth_steps = dict()
    step_count = 0
    while end_flag==1:

        # Step 4: Choose element with fewest boundary nodes
        if elem_indices[boundary_element_flag==0].shape[0] == 0:
            #  breakpoint()
             break
        
        seed_elem = np.random.choice(elem_indices[boundary_element_flag==0],size=1)[0]

        # Step 5: Domain growth
        domain = set([seed_elem])
        frontier = {seed_elem}
        
         
        end_flag_2 = True
        while end_flag_2:
            covered_elements = list(frontier)
            
            boundary_element_flag[covered_elements]+=1
            # Get all unique nodes in the frontier
            frontier_nodes = set()
            for eid in frontier:
                frontier_nodes.update(elements_int[eid])

            # Get neighboring elements of those nodes
            neighbor_elems = set()
            for nid in frontier_nodes:
                neighbor_elems.update(node_to_elems[nid])
            
            new_elems = neighbor_elems - domain
            
            domain.update(new_elems)

            frontier = set()
            possibilities = set()
            for eid in domain:
                if boundary_element_flag[eid]==0:
                    possibilities.update({eid})
            if len(possibilities)==0:
                end_flag_2=False
            else:
                temp = list(possibilities)
                frontier = {temp[0]}

        print(f"Seed element index: {seed_elem}")
        print(f"Total elements in grown domain: {len(domain)}")
        total_domain.update(domain)
        tempp = element_global_indices[int_point_flag]
         
        element_global_islands[tempp[list(domain)]] = no_of_islands+1
        no_of_islands+=1
        print("no of islands:", no_of_islands)
         
        
        if len(total_domain)>=elements_int.shape[0]:
            end_flag = 0

        # --- Save using NumPy ---
    machineeps = 1e-9
    if np.abs(np.min(element_global_islands))<machineeps:
        element_global_islands+=1
    os.makedirs(folder_path, exist_ok=True)
     
    # Save element indices
    grown_elem_indices = np.array(sorted(total_domain), dtype=int)

    # Save full element rows
    grown_elem_rows = elements_np[grown_elem_indices]

    # Reload original DataFrame
    elements_df = pd.read_csv(f'{folder_path}/{loadstep}/output_elements.csv')

    # Add the island_id column (it’s already in the correct order)
    elements_df['island_id'] = element_global_islands.astype(int)

    # Overwrite the CSV
    elements_df.to_csv(f'{folder_path}/{loadstep}/output_elements.csv', index=False)

    print("'island_id' column written to output_elements.csv")

if __name__=='__main__':
    growIslands('TestMat')
