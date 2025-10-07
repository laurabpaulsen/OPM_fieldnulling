#%% Imports and global variables
import numpy as np
from glob import glob 
from scipy.optimize import minimize, nnls, lsq_linear, dual_annealing, differential_evolution
# from nulling_my_coils import comp_field_control, opm_fieldline_control, reset_baseline 
import mne
import matplotlib.pyplot as plt
import matplotlib
# base_vec = np.array([57.5, 106.9, 3.2, 0.0, 0.0, 0.0, 0.0, 0.0])
rescale_step = np.array([1, 1, 1, 0.15, 0.15, 0.15, 0.15, 0.15])

SENSORS = np.array([0,1,2,3,4,5,6,7]) 
MODULES = ['A','B','C','D','E','F','G','H']
AXIS = ['X','Y','Z'] 
sensor_selection = {m: {a: SENSORS+8*(i*len(AXIS)+j) for j,a in enumerate(AXIS)} for i,m in enumerate(MODULES)}

switch = False
if not switch:
    # Data from simulated nulling
    # data_dir = glob("..\\data\\*\\sim*.fif"); print(len(data_dir)) 
    # data_dir = glob("..\\data\\*\\nulling*.fif"); print(len(data_dir)) 
    data_dir = glob("..\\data\\*\\EmptyRoom*.fif"); print(len(data_dir)) 
    raw_array = list()
    for dir in data_dir:
        raw_array.append(mne.io.read_raw_fif(dir))
    # raw_array = mne.io.read_raw_fif(data_dir[0])
else:
    # Data from sensor status & data thread
    data_dir = glob("..\\data\\optim01_*.npz"); print(len(data_dir)) 
    raw_array = {'on': list(),'off': list()}
    for dir in data_dir:
        if "fieldzero_on" in dir:
            raw_array['on'].append(np.load(dir, allow_pickle=True))
        else:
            raw_array['off'].append(np.load(dir, allow_pickle=True))

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
#%% Viz my noise level please!!!

matplotlib.use('QtAgg')
# selective = np.array([17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
#                       42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64])-1
selective = np.array([17,18,19,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,42,43,44,45,46,48,49,52,54,56,58,60,61,62])
fig,ax = plt.subplots(2,sharex=True,height_ratios=[0.5,0.5]); txt = ['Middle','Quiet Spot']
for i in range(2):
    tmp = raw_array[i].compute_psd(picks=np.concatenate([selective+64*2]), fmin=1, fmax=80)
    # print(raw_array[i].filenames)
    tmp.plot(amplitude=True, dB=False, average=False, axes=ax[i])
    # ax[i].set_ylim(0,100)
    ax[i].set_title(txt[i])
    
plt.show()

#%%
mne.viz.set_browser_backend('qt')  # interactive
tmp = raw_array[i].compute_psd(picks=np.concatenate([selective + 64*2]), fmin=1, fmax=80)
fig = tmp.plot(amplitude=True, dB=False, average=False)  # no axes=

#%% DATA
mne.viz.set_browser_backend('qt')
# mne.viz.plot_raw(raw_array[0])
selective = np.array([18,22,24,26,27,28,32,34,35,38,41,44,56,57,61])-1 # 30 is gone??
raw_array[2].plot_psd(tmin=25,picks=np.concatenate([selective,selective+64,selective+64*2]),fmax=30,dB=False,estimate='amplitude')
#%% Comparing coils null and not-null frequency tagged data  
data = {'null':[],'notnull':[]}
freqs = {'null':[],'notnull':[]}
frequency_point = {'null':[],'notnull':[]}
for i,j in enumerate(['null','notnull']):
    freqtag = raw_array[i+1].compute_psd(tmin=25 ,fmax=30) # picks=np.concatenate([selective,selective+64,selective+2*64])
    freqtag.plot(amplitude=True,dB=False); plt.show()
    data[j], freqs[j] = freqtag.get_data(return_freqs=True)
    frequency_point[j] = [np.argmin(abs(freqs[j]-k)) for k in [2,3,5,7,11,13,17,19]]
print(np.max(data['null'][:,frequency_point['null']]-data['notnull'][:,frequency_point['notnull']]))

which_coil = 3; 
for which_coil in range(8):
    print(f'working on Coil: {ch_names[which_coil]}')
                        
    # plt.subplot(1,2,1).imshow(data['null'][:,frequency_point['null']])
    # plt.subplot(1,2,2).imshow(data['notnull'][:,frequency_point['notnull']])
    # plt.subplot(1,2,1).plot(data['null'][:,frequency_point['null'][which_coil]])
    # plt.subplot(1,2,2).plot(data['notnull'][:,frequency_point['notnull'][which_coil]])
    # plt.show()

    # plt.title(f'Coil: {ch_names[which_coil]}')
    plt.subplot(2,1,1).imshow(data['null'][:,frequency_point['null'][which_coil]].reshape(64,3)[9:19,:].T)
    plt.subplot(2,1,2).imshow(data['notnull'][:,frequency_point['notnull'][which_coil]].reshape(64,3)[9:19,:].T)

    plt.show()

#%% Comparing regular sequential coil tuning and frequency tagging data
data = {'regular':[],'notnull':[]}
freqs = {'regular':[],'notnull':[]}
frequency_point = {'regular':[],'notnull':[]}
bad_annot = mne.Annotations(
    onset=[0,2,7,15,21,30,38,46,55,63],
    duration=[1,4,7,5,8,7,7,8,7,7.80],
    description=['bad','bad','bad','bad','bad','bad','bad','bad','bad','bad']
)
# raw_array[0].set_annotations(bad_annot)
# FS = 750 # Sample frequency
FS = 375
jump_point = np.array([3.727,11.369,17.478,24.805,32.763,41.303,50.399,57.699]) # locations where the coils are changed in order
# frequency_point['regular'] = jump_point # I guess?
tmp,freqs['regular'] = raw_array[0].get_data(picks='data',return_times=True)
frequency_point['regular'] = np.array([np.argmin(abs(freqs['regular']-k)) for k in jump_point])
data['regular'] = abs(tmp[:,frequency_point['regular']-int(0.3*FS)] 
                     -tmp[:,frequency_point['regular']+int(0.3*FS)])


freqtag = raw_array[2].compute_psd(tmin=25 ,fmax=30,reject_by_annotation=True) # picks=np.concatenate([selective,selective+64,selective+2*64])
# freqtag.plot(amplitude=True,dB=False); plt.show()
data['notnull'], freqs['notnull'] = freqtag.get_data(return_freqs=True)
frequency_point['notnull'] = [np.argmin(abs(freqs['notnull']-k)) for k in [2,3,5,7,11,13,17,19]]
# print(np.max(data['null'][:,frequency_point['null']]-data['notnull'][:,frequency_point['notnull']]))
#%%
which_coil = 3; 
for which_coil in range(8):
    print(f'working on Coil: {ch_names[which_coil]}')
                        
    plt.subplot(1,2,1).imshow(data['null'][:,frequency_point['null']])
    plt.subplot(1,2,2).imshow(data['notnull'][:,frequency_point['notnull']])

    # plt.subplot(1,2,1).plot(data['null'][:,frequency_point['null'][which_coil]])
    # plt.subplot(1,2,2).plot(data['notnull'][:,frequency_point['notnull'][which_coil]])
    # plt.show()

    # plt.title(f'Coil: {ch_names[which_coil]}')
    # plt.subplot(2,1,1).imshow(data['regular'][:,which_coil].reshape(64,3)[39:49,:].T)
    # plt.subplot(2,1,2).imshow(data['notnull'][:,frequency_point['notnull'][which_coil]].reshape(64,3)[39:49,:].T)

    plt.show()

#%% Adapting data to comp alg
if not switch:
    data_array = raw_array[0].get_data()
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
else:
    which_set = 2; available_sets = [('off',0),('off',1),('on',0),('on',1)]
    keys = raw_array[available_sets[which_set][0]][available_sets[which_set][1]].files # name of the stored variables in the .npz file
    sensor_status = raw_array[available_sets[which_set][0]][available_sets[which_set][1]][keys[2]]#.reshape(9,-1) # sensor status information for each coil setting
    active_sensor = raw_array[available_sets[which_set][0]][available_sets[which_set][1]][keys[-1]] # which sensors were active in the session
    coil_settings = raw_array[available_sets[which_set][0]][available_sets[which_set][1]][keys[0]] # coil settings i.e. formerly known as baseline
    # data_array    = raw_array['off'][0][keys[1]] # the averaged data collected per coil setting ;;; The function to save the data has been stored instead of the data
    applied_fields = np.empty((len(active_sensor),3,9))
    for i,status_i in enumerate(sensor_status):
        applied_fields[:,:,i] = np.array([[float(status_i[key]['BFX']),
                                           float(status_i[key]['BFY']),
                                           float(status_i[key]['BFZ'])] 
                                           for key in status_i if status_i[key]["LLS"] == "1"])
    
    data = applied_fields.copy()


for i in range(8):
    if np.any((data[:,:,i]-data[:,:,i+1]) != 0):
        plt.imshow(data[:,:,i].T)
        plt.show()
    else:
        print('All "Applied fields" are identical!!')




#%%
if not switch:
    base_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0]); rescale_step = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # base values and rescale step for the comp coils
    baseline = np.concatenate((np.tril(np.ones((9,8)),-1)*rescale_step[:,np.newaxis] + base_vec, np.ones((9,1))),axis=1) # Initial guess
else:
    baseline = coil_settings.copy()

def dual_annealing_residuals(baseline,data,weight = [.5, .5, 1], is_penalty=True,default=True):
    def model_me_Lvec(baseline,data,weight):

        n_channels = data.shape[0]

        for i in range(9):
            data[:,:,i] = data[:,:,i] * weight

        L = np.empty((9, 3, n_channels))
        for k in range(n_channels): # number of sensors
            L[:,:,k] = np.linalg.lstsq(baseline,np.transpose(np.squeeze(data[k,:,:])),rcond=None)[0]
        Lvec = L.reshape(9,3*n_channels)
        offsets = Lvec[-1,:]
        Lvec = np.delete(Lvec, -1, axis=0)
        return Lvec, offsets
    Lvec, offsets = model_me_Lvec(baseline,data,weight)
    # rng = 10
    # bounds = [(baseline[0,i]-rng,baseline[0,i]+rng) for i in range(8)]    
    bounds = [(coil_dBdV[i]*-10/1e-9,coil_dBdV[i]*10/1e-9) for i in range(8)]#; print(bounds)
    # bounds = [(0,coil_dBdV[i]*10/1e-9) for i in range(8)]

    def obj_fun(baseline, Lvec, offsets,is_penalty):
        # squared error
        squared_error = np.sum((Lvec.T @ baseline + offsets)**2)

        # penalty for negative residuals
        penalty = np.sum((Lvec.T @ baseline + offsets) < 0) + 1

        if is_penalty:
        # print(squared_error*penalty)
            return squared_error*penalty
        else:
            return squared_error
        
    if default:
        result = dual_annealing(obj_fun,bounds,args=(Lvec,offsets,is_penalty),x0=baseline[0,:-1])
        # result = dual_annealing(obj_fun,bounds,args=(Lvec,offsets,is_penalty),x0=baseline[0,:-1],maxfun=10000000,maxiter=10000)
    else:
        result = differential_evolution(obj_fun,bounds,args=(Lvec,offsets,is_penalty),x0=baseline[0,:-1])
        # result = differential_evolution(obj_fun,bounds,args=(Lvec,offsets),x0=baseline[0,:-1],maxfun=10000000,maxiter=10000)
    
    # print(result.fun)
    return result

def nonneg_residual_lsq_algorithm(coil_settings, data, weight = [0.5, 0.5, 1.], is_penalty=True):
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
    
    def obj_val(baseline, Lvec, offsets, is_penalty):


        # squared error
        squared_error = np.sum((Lvec.T @ baseline + offsets)**2)

        # penalty for negative residuals
        penalty = np.sum((Lvec.T @ baseline + offsets) < 0) + 1

        if is_penalty:
            # print(penalty)
            return squared_error*penalty
        else:
            return squared_error
    
    def constraint(baseline,Lvec, offsets):
        return np.sum((Lvec.T @ baseline + offsets) < 0)

    # new_coil_settings = minimize(obj_val, coil_settings[0,:-1], args=(Lvec,offsets), constraints={'type': 'eq', 'fun': constraint, 'args': (Lvec,offsets)})
    new_coil_settings = minimize(obj_val, coil_settings[0,:-1], args=(Lvec, offsets, is_penalty), method="Nelder-Mead")#, constraints={'type': 'eq', 'fun': constraint, 'args': (Lvec, offsets)})

    return new_coil_settings

def bound_lsq_algorithm(baseline,data, weight=[0.5,0.5,1],lam=0.0):
    

    n_channels = data.shape[0]
    n_coil_settings = coil_settings.shape[0]
    n_coil_parameters = coil_settings.shape[1]

    for i in range(9):
        data[:,:,i] = data[:,:,i] * weight

    L = np.empty((9, 3, n_channels))
    for k in range(n_channels): # number of sensors
        L[:,:,k] = np.linalg.lstsq(baseline,np.transpose(np.squeeze(data[k,:,:])),rcond=None)[0]
    Lvec = L.reshape(9,3*n_channels)
    offsets = Lvec[-1,:]
    Lvec = np.delete(Lvec, -1, axis=0).T

    lower = [coil_dBdV[i]*-10/1e-9 for i in range(8)]
    upper = [coil_dBdV[i]*10/1e-9 for i in range(8)]

    # Ridge/Tikhonov regularization to keep voltages reasonable (λ is small, tune as needed)
    # try 0.0 first; e.g., 0.1 * np.max(np.linalg.svd(Aw, compute_uv=False)) for gentle damping

    if lam > 0:
        # Augment for ridge: minimize ||Aw v + ow||^2 + lam^2 ||v||^2
        A_aug = np.vstack([Lvec, lam * np.eye(Lvec.shape[1])])
        b_aug = np.concatenate([-offsets, np.zeros(Lvec.shape[1])])
        res = lsq_linear(A_aug, b_aug, bounds=(lower, upper))
    else:
        res = lsq_linear(Lvec, -offsets, bounds=(lower, upper))

    # v_opt = res.x
    # print(f"coil settings:\t{np.round(v_opt, 4)}\t with objective value: {res.cost:.6g}")
    return res
               

no_weight = [.5,.5,1]
result_nonnegres = nonneg_residual_lsq_algorithm(baseline,data,weight=no_weight, is_penalty=True)
result_dual_anne = dual_annealing_residuals(baseline,data,weight=no_weight, is_penalty=True,default=True)
# results_boundlsq = bound_lsq_algorithm(baseline,data,weight=no_weight,lam=0.0)
print(f'coil settings: \t{np.round(result_nonnegres.x,4)}\t with objective value: {result_nonnegres.fun}')
print(f'coil settings: \t{np.round(result_dual_anne.x,4)}\t with objective value: {result_dual_anne.fun}')
# print(f'coil settings: \t{np.round(results_boundlsq.x,4)}\t with objective value: {results_boundlsq.cost}')
# print(result_dual_anne)
who_dat_sensor = np.array([ f'{i}{j+1}' for i in ['A','B','C','D','E','F','G','H'] for j in range(8)])
print(who_dat_sensor[active_sensor])