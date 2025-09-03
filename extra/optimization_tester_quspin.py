#%% Imports and global variables
import numpy as np
from glob import glob 
from scipy.optimize import minimize, nnls, lsq_linear, dual_annealing, differential_evolution
# from nulling_my_coils import comp_field_control, opm_fieldline_control, reset_baseline 
import mne
# base_vec = np.array([57.5, 106.9, 3.2, 0.0, 0.0, 0.0, 0.0, 0.0])
rescale_step = np.array([1, 1, 1, 0.15, 0.15, 0.15, 0.15, 0.15])


data_dir = glob('..\\New_nulling_quspin\\QuspinData\\sim*\\**.fif'); print(len(data_dir))
raw_array = mne.io.read_raw_fif(data_dir[0])

# Variables to understand and limit gradient coil settings
ch_names = ['Y','Z','X','dBy/dy','dBz/dy','dBz/dz','dBz/dx','dBy/dx']
coil_dBdI = np.array([0.4570,0.6372,0.9090,2.6369,2.6271,1.4110,4.8184,2.4500]) 
scales = np.array([1e-6,1e-6,1e-6,1e-6/100,1e-6/100,1e-6/100,1e-6/100,1e-6/100]) 
coil_dBdI = coil_dBdI*scales #in T/A(/cm) 
coil_R = np.array([19.52,13.93,12.16,18.48,16.67,14.18,10.39,8.51]) # Measured values, in Ohms
coil_dBdV = coil_dBdI / coil_R
min_voltage = -10
max_voltage = 10
def convert2volt(field,ch):
     return field / coil_dBdV[ch] * 1e-9

#%% DATA
mne.viz.set_browser_backend('qt')
mne.viz.plot_raw(raw_array)

#%% Adapting data to comp alg
data_array = raw_array.get_data()
FS = 750
pockets = np.array([[1,2],[6,7],[14,15],[20,21],[29,30],[37,38],[45,46],[54,55],[62,63]])*FS # pockets of data to average
data = np.empty((data_array.shape[0],9)) 

#- prep data for optimization -#
for idx,arr in enumerate(pockets):
    data[:,idx] = data_array[:,arr[0]:arr[1]].mean(axis=1)

data = data[:192].reshape(64,3,9)
selective = np.array([18,22,24,26,27,28,30,32,34,35,38,41,44,56,57,61])-1
data = data[selective,:,:] 
print(data.shape)


base_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0]); rescale_step = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # base values and rescale step for the comp coils
baseline = np.concatenate((np.tril(np.ones((9,8)),-1)*rescale_step[:,np.newaxis] + base_vec, np.ones((9,1))),axis=1) # Initial guess

def dual_annealing_residuals(baseline,data):
    def model_me_Lvec(baseline,data):

        n_channels = data.shape[0]

        weight = [.5, .5, 1]
        for i in range(9):
            data[:,:,i] = data[:,:,i] * weight

        L = np.empty((9, 3, n_channels))
        for k in range(n_channels): # number of sensors
            L[:,:,k] = np.linalg.lstsq(baseline,np.transpose(np.squeeze(data[k,:,:])),rcond=None)[0]
        Lvec = L.reshape(9,3*n_channels)
        offsets = Lvec[-1,:]
        Lvec = np.delete(Lvec, -1, axis=0)
        return Lvec, offsets
    Lvec, offsets = model_me_Lvec(baseline,data)
    # rng = 10
    # bounds = [(baseline[0,i]-rng,baseline[0,i]+rng) for i in range(8)]    
    bounds = [(coil_dBdV[i]*-10/1e-9,coil_dBdV[i]*10/1e-9) for i in range(8)]
    # bounds = [(0,coil_dBdV[i]*10/1e-9) for i in range(8)]

    def obj_fun(baseline, Lvec, offsets):
        # squared error
        squared_error = np.sum((Lvec.T @ baseline + offsets)**2)

        # penalty for negative residuals
        penalty = np.sum((Lvec.T @ baseline + offsets) < 0) + 1

        # print(squared_error*penalty)
        return squared_error*penalty
        

    result = dual_annealing(obj_fun,bounds,args=(Lvec,offsets),x0=baseline[0,:-1])
    # result = dual_annealing(obj_fun,bounds,args=(Lvec,offsets),x0=baseline[0,:-1],maxfun=10000000,maxiter=10000)
    # result = differential_evolution(obj_fun,bounds,args=(Lvec,offsets),x0=baseline[0,:-1])
    # result = differential_evolution(obj_fun,bounds,args=(Lvec,offsets),x0=baseline[0,:-1],maxfun=10000000,maxiter=10000)
    # print(result.fun)
    return result

def nonneg_residual_lsq_algorithm(coil_settings, data, weight = [0.5, 0.5, 1.]):
    """
    Potential problems with this approach:

    * CONTRAINTS FOR THE OPTIMISATION: As close to zero and on the same side of zero for each sensors (positive)

                * The idea is that this forces the optimisation also to minimise the gradients

                * Here the assumption is that all the sensors are pointing in the same direction

                * This would not work well with sensors that are pointing in opposite direction

                * One could run the optimisation on a set of sensors pointing in somewhat the same direction
    """
    # Check dimensions of coil_settings
    if coil_settings.ndim != 2:
        raise ValueError(f"coil_settings must be a 2D array, but got shape {coil_settings.shape}")

    # Check dimensions of data
    if data.ndim != 3:
        raise ValueError(f"data must be a 3D array, but got shape {data.shape}")

    # Extract dimensions
    n_channels = data.shape[0]
    n_coil_settings = coil_settings.shape[0]
    n_coil_parameters = coil_settings.shape[1]
    L = np.empty((n_coil_parameters, 3, n_channels))
    if weight:
        for i in range(n_coil_settings):
            data[:,:,i] = data[:,:,i] * weight

    # Compute least squares
    for k in range(n_channels):
        L[:, :, k] = np.linalg.lstsq(coil_settings, np.transpose(np.squeeze(data[k, :, :])), rcond=None)[0]
    Lvec = L.reshape(n_coil_parameters, 3 * n_channels)
    offsets = Lvec[-1,:]
    Lvec = np.delete(Lvec, -1, axis=0)
    
    def obj_val(baseline, Lvec, offsets):


        # squared error
        squared_error = np.sum((Lvec.T @ baseline + offsets)**2)

        # penalty for negative residuals
        penalty = np.sum((Lvec.T @ baseline + offsets) < 0) + 1

        # print(penalty)
        return squared_error*penalty
    
    def constraint(baseline,Lvec, offsets):
        return np.sum((Lvec.T @ baseline + offsets) < 0)

    # new_coil_settings = minimize(obj_val, coil_settings[0,:-1], args=(Lvec,offsets), constraints={'type': 'eq', 'fun': constraint, 'args': (Lvec,offsets)})
    new_coil_settings = minimize(obj_val, coil_settings[0,:-1], args=(Lvec, offsets), method="Nelder-Mead")#, constraints={'type': 'eq', 'fun': constraint, 'args': (Lvec, offsets)})

    return new_coil_settings

result = dual_annealing_residuals(baseline,data)
print(result)
result = nonneg_residual_lsq_algorithm(baseline,data)
print(result)