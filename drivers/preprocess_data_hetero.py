#=====================================================================
# INITIALIZATIONS:
#=====================================================================
#torch
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

#numpy
import numpy as np
np.random.seed(0)

#matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#scipy
import scipy
from scipy import sparse

#pandas
import pandas as pd

#others:
import os
import sys
sys.path.insert(1, '../core/')
from contextlib import contextmanager
import shutil
from itertools import combinations
from distutils.dir_util import copy_tree

#custom:
from data_definitions import *
from load_fem_data_wedge_2intpoints import *
from utilities import *


fem_material = sys.argv[1]
noise = sys.argv[2]

    

def extractSystemOfEquations(fem_dir, loadsteps, cluster_values, noiseLevel):
    """
    Processes nodal and element data at various load steps to assemble the linear equation
    [A1;lambda_r*A2]*theta = [b1;lambda_r*b2]

    _Input Arguments_
    - `fem_dir` - File path location for data from FEM simulations
    - `loadsteps` - The deformation steps from the FEM simulations used to discover the material properties
    """
    
    cluster_values = np.concatenate((cluster_values,cluster_values))
    cluster_values_old = cluster_values.copy()
    # Determine the number of clusters
    num_clusters = np.max(cluster_values)
    # breakpoint()
    for loadstep in loadsteps:
        print(f"Processing loadstep: {loadstep}")

        # Load FEM data
        data = loadFemData(fem_dir + '/' + str(loadstep), AD=True, noiseLevel=noiseLevel, noiseType='displacement')

        d_features_dI1 = data.featureSet.d_features_dI1
        d_features_dI2 = data.featureSet.d_features_dI2
        d_features_dI3 = data.featureSet.d_features_dI3
        d_features_dIa = data.featureSet.d_features_dIa
        d_features_dIb = data.featureSet.d_features_dIb

        dI1dF = data.dI1dF
        dI2dF = data.dI2dF
        dI3dF = data.dI3dF
        dIadF = data.dIadF
        dIbdF = data.dIbdF

        # Assemble the feature derivatives w.r.t. F
        d_features_dF11 = d_features_dI1 * dI1dF[:,0:1] + d_features_dI2 * dI2dF[:,0:1] + d_features_dI3 * dI3dF[:,0:1] + d_features_dIa * dIadF[:,0:1] + d_features_dIb * dIbdF[:,0:1]
        d_features_dF12 = d_features_dI1 * dI1dF[:,1:2] + d_features_dI2 * dI2dF[:,1:2] + d_features_dI3 * dI3dF[:,1:2] + d_features_dIa * dIadF[:,1:2] + d_features_dIb * dIbdF[:,1:2]
        d_features_dF21 = d_features_dI1 * dI1dF[:,2:3] + d_features_dI2 * dI2dF[:,2:3] + d_features_dI3 * dI3dF[:,2:3] + d_features_dIa * dIadF[:,2:3] + d_features_dIb * dIbdF[:,2:3]
        d_features_dF22 = d_features_dI1 * dI1dF[:,3:4] + d_features_dI2 * dI2dF[:,3:4] + d_features_dI3 * dI3dF[:,3:4] + d_features_dIa * dIadF[:,3:4] + d_features_dIb * dIbdF[:,3:4]
        d_features_dF13 = d_features_dI1 * dI1dF[:,4:5] + d_features_dI2 * dI2dF[:,4:5] + d_features_dI3 * dI3dF[:,4:5] + d_features_dIa * dIadF[:,4:5] + d_features_dIb * dIbdF[:,4:5]
        d_features_dF23 = d_features_dI1 * dI1dF[:,5:6] + d_features_dI2 * dI2dF[:,5:6] + d_features_dI3 * dI3dF[:,5:6] + d_features_dIa * dIadF[:,5:6] + d_features_dIb * dIbdF[:,5:6]
        d_features_dF31 = d_features_dI1 * dI1dF[:,6:7] + d_features_dI2 * dI2dF[:,6:7] + d_features_dI3 * dI3dF[:,6:7] + d_features_dIa * dIadF[:,6:7] + d_features_dIb * dIbdF[:,6:7]
        d_features_dF32 = d_features_dI1 * dI1dF[:,7:8] + d_features_dI2 * dI2dF[:,7:8] + d_features_dI3 * dI3dF[:,7:8] + d_features_dIa * dIadF[:,7:8] + d_features_dIb * dIbdF[:,7:8]
        d_features_dF33 = d_features_dI1 * dI1dF[:,8:9] + d_features_dI2 * dI2dF[:,8:9] + d_features_dI3 * dI3dF[:,8:9] + d_features_dIa * dIadF[:,8:9] + d_features_dIb * dIbdF[:,8:9]


        num_nodes_per_element = len(data.gradNa)
        dim = data.x_nodes.shape[1]

        # Prepare to assemble A_nodes
        A_nodes = torch.zeros(data.numNodes, dim, getNumberOfFeatures())
        # A_clusters = [torch.zeros_like(A_nodes) for _ in range(num_clusters)]
        A_clusters = []
        # Group derivatives for ease of iteration
        d_features_dF = [
            d_features_dF11, d_features_dF12, d_features_dF13,
            d_features_dF21, d_features_dF22, d_features_dF23,
            d_features_dF31, d_features_dF32, d_features_dF33
        ]
  

        #breakpoint()
        
        a = cluster_values
        b = np.arange(np.max(cluster_values))+1
        c = (a.reshape(-1,1)==b.reshape(1,-1))
        d = np.sum(c, axis=0)
        max_cluster = np.argmax(d)+1
        scaleup =1
        for cluster in range(1,int(np.max(cluster_values))+1):
            # breakpoint()
            cluster_indices = (cluster_values == cluster)

            A_cluster = torch.zeros_like(A_nodes)
            for a in range(num_nodes_per_element):
                lhs1 = d_features_dF11 * data.gradNa[a][:, 0:1] + d_features_dF12 * data.gradNa[a][:, 1:2] + d_features_dF13 * data.gradNa[a][:, 2:3]
                lhs1 = lhs1 * data.qpWeights.unsqueeze(-1)

                lhs2 = d_features_dF21 * data.gradNa[a][:, 0:1] + d_features_dF22 * data.gradNa[a][:, 1:2] + d_features_dF23 * data.gradNa[a][:, 2:3]
                lhs2 = lhs2 * data.qpWeights.unsqueeze(-1)

                lhs3 = d_features_dF31 * data.gradNa[a][:, 0:1] + d_features_dF32 * data.gradNa[a][:, 1:2] + d_features_dF33 * data.gradNa[a][:, 2:3]
                lhs3 = lhs3 * data.qpWeights.unsqueeze(-1)

                lhs = torch.stack((lhs1, lhs2, lhs3), dim=1)
    
                A_cluster.index_add_(0, data.connectivity[a][(cluster_indices)].squeeze(), lhs[(cluster_indices)])

            if cluster<np.max(cluster_values):
                A_clusters.append(scaleup*A_cluster)
            else:
                A_clusters.append(A_cluster)   

        A_dofs = torch.column_stack(
            [torch.flatten(A_cluster, start_dim=0, end_dim=1) for A_cluster in A_clusters]
        )
        # Create b_nodes based on mass type
        mass_type = get_mass_type()
        # breakpoint()
        if mass_type == 'lumped':
            b_nodes = -scaleup*data.lumped_mass_acceleration
        else:
            raise ValueError('Incorrect mass_type option')
        
        b_dofs = torch.flatten(b_nodes.clone(), start_dim=0, end_dim=1)

        dirichlet_dofs = torch.flatten(data.dirichlet_nodes.clone(), start_dim=0, end_dim=1)
        A_free = A_dofs[(~dirichlet_dofs),:]#&(temp_mask_dofs==0)
        
        b_free = b_dofs[(~dirichlet_dofs)]#&(temp_mask_dofs==0)

        #Assemble fixed-dof equations
        A_fix = torch.zeros(len(data.reactions), np.max(cluster_values)*getNumberOfFeatures())
        b_fix = torch.zeros(len(data.reactions))


        for r in range(len(data.reactions)):

            bc_dofs = torch.flatten(data.reactions[r].dofs.clone(),start_dim=0,end_dim=1)
            #breakpoint()
            A_fix[r,:] = torch.sum(A_dofs[bc_dofs,:],dim=0)

            b_fix[r] = torch.sum(b_dofs[bc_dofs],dim=0) + data.reactions[r].force
        b_fix = scaleup*b_fix
        
        W = data.featureSet.features
        num_features = getNumberOfFeatures()
        # list_features = torch.arange(num_features)
        W_new = torch.zeros(W.shape[0],num_clusters*num_features)
        for j in range(num_clusters):
            mask_j = (cluster_values == (j+1))
            # breakpoint()
            W_new[mask_j,(j*num_features):(j+1)*num_features] = W[mask_j,:]
        headerstring = ""
        fmtstring = ""
        for i in range(num_clusters):
            for j in range(num_features):
                headerstring+='Feat'+str(i)+str(j)
                fmtstring+='%20.20f'
                if (i<num_clusters-1)|(j<num_features-1):
                    headerstring+=','
                    fmtstring+=','
        #Extracting position information for free-dofs only
        dof_x = torch.flatten(torch.cat((data.x_nodes[:,0:1],data.x_nodes[:,0:1],data.x_nodes[:,0:1]),dim=1),start_dim=0,end_dim=1)
        dof_y = torch.flatten(torch.cat((data.x_nodes[:,1:2],data.x_nodes[:,1:2],data.x_nodes[:,1:2]),dim=1),start_dim=0,end_dim=1)
        dof_z = torch.flatten(torch.cat((data.x_nodes[:,2:3],data.x_nodes[:,2:3],data.x_nodes[:,2:3]),dim=1),start_dim=0,end_dim=1)

        dof_x = dof_x[~dirichlet_dofs]
        dof_y = dof_y[~dirichlet_dofs]
        dof_z = dof_z[~dirichlet_dofs]

        np.savetxt(fem_dir+'/'+str(loadstep)+'/A1.csv', A_free.cpu().detach().numpy(), delimiter=",")
        np.savetxt(fem_dir+'/'+str(loadstep)+'/b1.csv', b_free.cpu().detach().numpy(), delimiter=",")

        np.savetxt(fem_dir+'/'+str(loadstep)+'/A2.csv', A_fix.cpu().detach().numpy(), delimiter=",")
        np.savetxt(fem_dir+'/'+str(loadstep)+'/b2.csv', b_fix.cpu().detach().numpy(), delimiter=",")

        np.savetxt(fem_dir+'/'+str(loadstep)+'/dof_x.csv', dof_x.cpu().detach().numpy(), delimiter=",")
        np.savetxt(fem_dir+'/'+str(loadstep)+'/dof_y.csv', dof_y.cpu().detach().numpy(), delimiter=",")
        np.savetxt(fem_dir+'/'+str(loadstep)+'/dof_z.csv', dof_z.cpu().detach().numpy(), delimiter=",")
        
        np.savetxt(fem_dir + '/' + str(loadstep) + '/features.csv', W_new.cpu().detach().numpy(), delimiter=",",header=headerstring,comments='')
        
        # breakpoint()
