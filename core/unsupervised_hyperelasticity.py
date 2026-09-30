import os
import sys
# sys.path.insert(1, '../drivers/')
sys.path.insert(2, '../core/')
import numpy as np
import scipy.stats as stats
from core_spike_slab import *
from helper import *
from features_library import *
from config import *
import pandas as pd
from features_library import getNumberOfFeatures

class RawData:
    def __init__(self, A1, b1, A2, b2, dof_x, dof_y, dof_z):
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        self.dof_x = dof_x
        self.dof_y = dof_y
        self.dof_z = dof_z


def check_data_name_validity(fem_material,noise_level):
    """

    """
    material_list = ['NeoHookeanJ2', 'NeoHookeanJ4', 'Isihara', 'HainesWilson', 'GentThomas','ArrudaBoyce','Ogden','Ogden3','Anisotropy','PDMS','Holzapfel']
    noise_list = ['none','low','high']
    # if fem_material not in material_list:
    #     raise ValueError('Incorrect fem_material provided')
    # if noise_level not in noise_list:
    #     raise ValueError('Incorrect noise_level provided')
    print('fem_material: ',fem_material)
    print('noise_level: ',noise_level)


def get_data(rng, fem_dir, prefix, fem_material, noise_level, loadstep, feature_filter):
    """

    """

    prefix = fem_dir+'/'+fem_material;
    if(prefix[-1]=='/'):
        prefix = prefix[0:-1]

    fem_path = prefix+'/'+str(loadstep)+'/';

    A1 = np.loadtxt(fem_path+'A1.csv', dtype = np.float64, delimiter=",");
    b1 = np.loadtxt(fem_path+'b1.csv', dtype = np.float64, delimiter=",");

    A2 = np.loadtxt(fem_path+'A2.csv', dtype = np.float64, delimiter=",");
    b2 = np.loadtxt(fem_path+'b2.csv', dtype = np.float64, delimiter=",");

    dof_x = np.loadtxt(fem_path+'dof_x.csv', dtype = np.float64, delimiter=",");
    dof_y = np.loadtxt(fem_path+'dof_y.csv', dtype = np.float64, delimiter=",");
    dof_z = np.loadtxt(fem_path+'dof_z.csv', dtype = np.float64, delimiter=",");
     
    # if feature_filter:
        # A1 = A1[:,feature_filter]
        # A2 = A2[:,feature_filter]
     
    return RawData(A1, b1, A2, b2, dof_x, dof_y, dof_z)

def filter_raw_data(rng, raw_data, filter_value, flaggedRows=True, step_folder='none'):
    """
    Randomly sub-sampling `filter_value` degrees of freedom from the data available at all quadrature points in `raw_data`

    _Input Arguments_

    -`rng` - Random number generator

    -`raw_data` - Contains `D` (derivatives of feature library) and `y` (inertia terms and reaction forces) evaluated at all quadrature points

    -`filter_value` - Number of degrees of freedom subsampled from the original data

    _Output Arguments_

    - `RawData(A1, b1, A2, b2, dof_x, dof_y)` - Object of class `RawData` which is constructed from subsampled data

    ---

    """

    if not filter_value:
        #filter_value is empty
        return raw_data

    A1 = raw_data.A1
    b1 = raw_data.b1

    A2 = raw_data.A2
    b2 = raw_data.b2

    residual_force_file_2 = os.path.join(step_folder, 'residuals_free_step_2.csv')
    residual_force_data_2 = pd.read_csv(residual_force_file_2)
    res_force_free_2 = residual_force_data_2[['res_free_norm_2']].values.squeeze()
    flaggedRows_sum = np.sum(flaggedRows,axis=1,dtype=bool)
    highlightFlags = np.zeros((flaggedRows.shape[0],),dtype=bool)
    
    A1blist = []
    b1blist = []
    for k in range(flaggedRows.shape[1]):
        if res_force_free_2[flaggedRows[:,k]].shape[0]==0:
            continue
        res_flagged = res_force_free_2[flaggedRows[:,k]]
        sort_ind = np.argsort(res_flagged)
        res_mean = np.mean(res_force_free_2[flaggedRows[:,k]])
        res_std = np.std(res_force_free_2[flaggedRows[:,k]])
        res_max = np.max(res_force_free_2[flaggedRows[:,k]])
         
        c1 = int(np.ceil(0.2*sort_ind.shape[0]))
        num_indices = np.arange(highlightFlags.shape[0])
        num_indices = num_indices[flaggedRows[:,k]]
        A1bounds = A1[flaggedRows[:,k],:]
        b1bounds = b1[flaggedRows[:,k]]
        
        num_indices = num_indices[sort_ind]
        A1bounds = A1bounds[sort_ind,:]
        b1bounds = b1bounds[sort_ind]
        
        num_indices = num_indices[:c1]
        A1bounds = A1bounds[:c1,:]
        b1bounds = b1bounds[:c1]
       
        highlightFlags[num_indices] = True
        A1blist.append(A1bounds)
        b1blist.append(b1bounds)
     
    A1bounds = np.row_stack(A1blist)
    b1bounds = np.concatenate(b1blist)
    
    numfeatures = getNumberOfFeatures()
    A1flist = []
    b1flist = []
    machineeps = 1e-8
    for k in range(A1.shape[1]//numfeatures):
        num_indices = np.arange(highlightFlags.shape[0])
        cluster_rows = A1[:,k*numfeatures:(k+1)*numfeatures]
        cluster_rows = np.sum((np.abs(cluster_rows)>machineeps),axis=1,dtype=bool)
        num_indices = num_indices[cluster_rows & (~flaggedRows_sum)]
        A1free = A1[cluster_rows & (~flaggedRows_sum),:]
        b1free = b1[cluster_rows & (~flaggedRows_sum)]
        idx2 = np.arange(A1free.shape[0]//3)
        idx2 = rng.permutation(idx2)
        c2 = int(np.ceil(0.02*idx2.shape[0]))
        temp_A1_1 = A1free[3*idx2[:c2],:]
        temp_b1_1 = b1free[3*idx2[:c2]]
        
        temp_A1_2 = A1free[3*idx2[:c2]+1,:]
        temp_b1_2 = b1free[3*idx2[:c2]+1]
        
        temp_A1_3 = A1free[3*idx2[:c2]+2,:]
        temp_b1_3 = b1free[3*idx2[:c2]+2]
        
        total_selections = np.concatenate((3*idx2[:c2],3*idx2[:c2]+1,3*idx2[:c2]+2))
        num_indices = num_indices[total_selections]
        highlightFlags[num_indices] = True
        A1flist.append(np.row_stack((temp_A1_1,temp_A1_2,temp_A1_3)))
        b1flist.append(np.concatenate((temp_b1_1,temp_b1_2,temp_b1_3)))
     
    A1free = np.row_stack(A1flist)
    b1free = np.concatenate(b1flist)

    dof_x = raw_data.dof_x
    dof_y = raw_data.dof_y
    dof_z = raw_data.dof_z
    
    arr_idx = np.arange(A1.shape[0])
    arr_idx_1 = arr_idx[flaggedRows_sum]
    arr_idx_2 = arr_idx[~flaggedRows_sum]
    
    
    dof_x = dof_x[np.concatenate((arr_idx_1,arr_idx_2))]
    dof_y = dof_y[np.concatenate((arr_idx_1,arr_idx_2))]
    dof_z = dof_z[np.concatenate((arr_idx_1,arr_idx_2))]
    
    A1 = np.row_stack((A1free,A1bounds))
    b1 = np.concatenate((b1free,b1bounds))
    
    file_highlights = pd.DataFrame({'Highlights': highlightFlags})
    file_highlights.to_csv(f'{step_folder}/Highlights.csv',index=False)
     
    return RawData(A1, b1, A2, b2, dof_x, dof_y, dof_z)

    
def process_raw_data(raw_data_set, lambda_r):
    """

    """

    D = raw_data_set[0].A1
    y = raw_data_set[0].b1
     
    # Careful: start iteration from 1
    for i in range(1,len(raw_data_set)):
        D = np.concatenate((D, raw_data_set[i].A1), axis=0)
        y = np.concatenate((y, raw_data_set[i].b1), axis=0)

    # Careful: start iteration from 0
    for i in range(0,len(raw_data_set)):
        # D = np.concatenate((D, lambda_r * raw_data_set[i].A2), axis=0)
        # y = np.concatenate((y, lambda_r * raw_data_set[i].b2), axis=0)
        D = np.concatenate((D, lambda_r * raw_data_set[i].A2[[1,4,6,9],:]), axis=0)
        y = np.concatenate((y, lambda_r * raw_data_set[i].b2[[1,4,6,9]]), axis=0)

    return Data(D, y)



def predict_energy_path(chain, theta_gt, fem_mat, feature_filter, deformation):
    """
    Predicts the energy deformation evolution for predicted and true material models along six deformation paths:
    i.) uniaxial tension, ii.) uniaxial compression, iii.) simple shear, iv.) biaxial tension, v.) biaxial compression, vi.) pure shear

    _Input Arguments_

    -`chain` - Object of class `Chain` (see `core_spike_slab` file)

    -`theta_gt` - The true set of feature coefficients for the benchmark material

    -`fem_mat` - The name of the benchmark material to be tested

    -`feature_filter` - The list of features to retain for constructing the Markov chain. Suppressed features will be highlighted with a red patch in the plot

    -`deformation` - Name of deformation path to be evaluated

    _Output Arguments_

    -`gamma` - Numpy array of different values for the deformation parameter

    -`energy_mean` - Energy corresponding to mean value of `theta` (feature coefficients) across different members of the Markov chain

    -`energy_plus` - 97.5 percentile energy branch across different members of the Markov chain

    -`energy_minus` - 2.5 percentile energy branch across different members of the Markov chain

    -`energy_gt` - Energy corresponding to the true feature coefficients

    -`energy` - Numpy array containing energy of all chain members at all deformation parameters

    ---

    """

    gamma = np.linspace(0.,1.,num=50)

    F11 = np.zeros_like(gamma) + 1.
    F12 = np.zeros_like(gamma)
    F21 = np.zeros_like(gamma)
    F22 = np.zeros_like(gamma) + 1.
    F13 = np.zeros_like(gamma)
    F31 = np.zeros_like(gamma)
    F32 = np.zeros_like(gamma)
    F23 = np.zeros_like(gamma)
    F33 = np.zeros_like(gamma) + 1.

    if deformation == 'tension':
        F11 = 1.+gamma

    elif deformation == 'simple_shear':
        F12 = gamma

    elif deformation == 'pure_shear':
        F11 = 1.+gamma
        F22 = 1./(1.+gamma)

    elif deformation == 'compression':
        F11 = 1./(1.+gamma)

    elif deformation == 'biaxial_tension':
        F11 = 1.+gamma
        F22 = 1.+gamma

    elif deformation == 'biaxial_compression':
        F11 = 1./(1.+gamma)
        F22 = 1./(1.+gamma)

    else:
        raise ValueError('Wrong input for deformation path')

    features = computeFeatures_numpy(F11, F12, F21, F22, F13, F23, F31, F32, F33)
    features_temp = features ##Remove if not working
    features = features[:,feature_filter]

    energy = features_temp @ chain.theta.T

    energy_mean = np.mean(energy,axis=1)
    energy_std = np.std(energy,axis=1)

    energy_plus  = np.percentile(energy, 97.5, axis=1)
    energy_minus = np.percentile(energy, 2.5, axis=1)

    energy_gt = features_temp @ (get_theta_gt(fem_mat,feature_filter)) ##Remove if not working

    return gamma, energy_mean, energy_plus, energy_minus, energy_gt, energy

