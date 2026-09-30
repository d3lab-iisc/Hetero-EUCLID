import numpy as np
import torch
torch.set_default_dtype(torch.float64)

# Random number seed
seed = 1234

# interactive job
interactive_job = False

suppress_chain_run = True

# FEM Data specifications
fem_dir = '..';# By default, the data is chosen to be quasistatic, unless specified in 3rd input argument.
fig_dir = '../drivers/plots';
fem_res = '1k'

# data params
lambda_r = 1.0
filter_value = [] # The number of degrees of freedom we wish to subsample to. Use [] for no filtering

# HSS hyper parameters
a_v = 0.5
b_v = 0.5
a_p = 0.1
b_p = 3.0
a_sigma = 1.
b_sigma = 1.

# chain params
chain_length = 500
burn = int(chain_length/4) #Only 75 percent of the original chain is retained.
parallel_chains = 3

alpha = 50.0* 3.14159/180.0

theta_fiber = alpha  # Direction of assumed fiber direction in feature library (In radians)
theta_fiber2= -alpha  # Direction of another assumed fiber direction in feature library (In radians)

a1 = np.cos(theta_fiber)
a2 = np.sin(theta_fiber)
b1 = np.cos(theta_fiber2)
b2 = np.sin(theta_fiber2)

#Vector a is the direction of the fiber a1 is x-component and a2 is y-component
a_mult = torch.zeros(9,1)
b_mult = torch.zeros(9,1)
a_mult[0:1,0] = a1**2.0
a_mult[1:2,0] = a1*a2
a_mult[2:3,0] = a1*a2
a_mult[3:4,0] = a2**2.0

b_mult[0:1,0] = b1**2.0
b_mult[1:2,0] = b1*b2
b_mult[2:3,0] = b1*b2
b_mult[3:4,0] = b2**2.0


def get_loadsteps(fem_material):
	#By default, only loadsteps 10, 20, 30, 40, 50 are considered. If required, different number of loadsteps can be considerd for specific materials by adjusting below.
	loadsteps = [1,2,3,4,5,6]
	return loadsteps

def get_feature_filter(fem_material):
	"""
	Certain features from the 28 features in the library can be suppressed by being excluded
	from the 'feature_filter' list below
	"""
	# feature_filter = range(12); # Features 27 and 28 (True Holzapfel features) are suppressed here.
	feature_filter = [0,1,2,3]
	return feature_filter
