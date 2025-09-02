
import numpy as np
from scipy.optimize import nnls, minimize



def apply_weights(data, weight):
    """Applies weights to data if provided."""
    for i in range(data.shape[2]):
        data[:,:,i] = data[:,:,i] * weight
    
    return data

def validate_inputs(coil_settings, data):
    """Validates the input dimensions for coil settings and data arrays."""
    if coil_settings.ndim != 2:
        raise ValueError(f"coil_settings must be a 2D array, but got shape {coil_settings.shape}")
    if data.ndim != 3:
        raise ValueError(f"data must be a 3D array, but got shape {data.shape}")
    if data.shape[2] != coil_settings.shape[0]:
        raise ValueError(f"Mismatch: data.shape[2] ({data.shape[2]}) should match coil_settings.shape[0] ({coil_settings.shape[0]})")



def nonneg_double_lsq_algorithm(coil_settings, data, weight=[0.5, 0.5, 1.]):
    """
    Performs a non-negative least squares (NNLS) algorithm to optimize coil values.

    Problem with this way of doing things:
    * forces the coils settings to be above zero
    

    Parameters
    ----------
    coil_settings : np.ndarray
        A (n_coil_settings, n_coil_parameters) matrix defining coil configurations.
    
    data : np.ndarray
        A (n_channels, 3, n_coil_settings) array containing measured field data.
    
    weight : list[float] | False, optionaln_channels
        A weight factor applied to the data. Defaults to [0.5, 0.5, 1.0] (higher weight on z-direction).

    Returns
    -------
    new_coil_values : np.ndarray
        Optimized coil values.
    
    residuals : float
        Residuals from the NNLS optimization.

    Raises
    ------
    ValueError
        If input dimensions are incorrect.
    """
    print("Using an old function that is probably not correct!!! forces the coil settings to be above 0")

    validate_inputs(coil_settings, data)

    # Extract dimensions
    n_channels = data.shape[0]
    n_coil_parameters = coil_settings.shape[1]


    # initialize L matrix
    L = np.empty((n_coil_parameters, 3, n_channels))

    # apply weighting if provided
    data = apply_weights(data, weight)

    # Compute least squares solution
    for k in range(n_channels):
        ch_data = np.transpose(data[k, :, :]) # extracting the data for the given sensor (x, y, z fields at each of the coil settings)
        x, residuals, rank, s =  np.linalg.lstsq(coil_settings, ch_data, rcond=None)
        L[:, :, k] = x

        
    # Reshape L and compute NNLS
    Lvec = L.reshape(n_coil_parameters, 3 * n_channels)
    offsets = Lvec[-1, :]
    Lvec = np.delete(Lvec, -1, axis=0)
    new_coil_values, residuals = nnls(np.transpose(Lvec), -offsets)

    print(f"Residuals: {residuals}")

    return new_coil_values


def nonneg_residual_lsq_algorithm(coil_settings, data, weight = [0.5, 0.5, 1.]) -> np.ndarray:
    """
    Potential problems with this approach:
    * We want residuals to be as close to zero and on the same side of zero for each sensors (positive)
		* The idea is that this forces the optimisation also to minimise the gradients
	* Here the assumption is that all the sensors are pointing in the same direction
	* This would not work well with sensors that are pointing in opposite direction
	* One could run the optimisation on a set of sensors pointing in somewhat the same direction

    Parameters:
    ----------
    coil_settings : ndarray (shape: M x N)
        A 2D array representing the coil settings, where M is the number of different settings and N is the number of parameters.
    
    data : ndarray (shape: C x M x 3)
        A 3D array containing measured data for C channels across M coil settings in 3 spatial directions (x, y, z).
    
    weight : list or ndarray, optional (default: [0.5, 0.5, 1.])
        A weight vector applied to the data to scale its importance across different dimensions of the data. By default the last dimension (z), which is the recording direction is weighted more. 

    Returns:
    -------
    new_coil_settings : ndarray
        The optimized coil settings obtained by minimizing the objective function.

    Potential Limitations:
    ---------------------
    * Assumes all sensors point in the same direction, which may not hold for all setups.
        * If sensors are oriented in opposite directions, the approach might fail.
    * Might be improved by running optimization separately for sensors with similar orientations.
    """

    # Validate dimensions
    validate_inputs(coil_settings, data)

    # Extract dimensions
    n_channels = data.shape[0]
    n_coil_parameters = coil_settings.shape[1]

    L = np.empty((n_coil_parameters, 3, n_channels))

    data = apply_weights(data, weight)

    for k in range(n_channels):
        ch_data = np.transpose(data[k, :, :]) # extracting the data for the given sensor (x, y, z fields at each of the coil settings)
        x, residuals, rank, s =  np.linalg.lstsq(coil_settings, ch_data, rcond=None)
        L[:, :, k] = x 
    
    Lvec = L.reshape(n_coil_parameters, 3 * n_channels)

    offsets = Lvec[-1,:]

    Lvec = np.delete(Lvec, -1, axis=0)

    def obj_function(coil_settings, Lvec, offsets):
        """
        Computes the objective function: squared error penalized by negative residuals.
        """     
        residuals = Lvec.T @ coil_settings + offsets
        squared_error = np.sum(residuals**2)
        penalty = np.sum(residuals < 0)  # Penalize negative residuals
 
        return squared_error * penalty
    
    new_coil_settings = minimize(obj_function, coil_settings[0,:-1], args=(Lvec, offsets), method="Nelder-Mead")


    return new_coil_settings.x



def sarangs_double_lsq_algorithm(coil_settings, data, weight = [0.5, 0.5, 1.]):

    validate_inputs(coil_settings, data)

    n_channels = data.shape[0]
    n_coil_parameters = coil_settings.shape[1]

    L = np.empty((n_coil_parameters, 3, n_channels))

    data = apply_weights(data, weight)

    for k in range(n_channels):
        L[:,:,k] = np.linalg.lstsq(coil_settings, np.transpose(np.squeeze(data[k,:,:])), rcond=None)[0]
    
    Lvec = L.reshape(n_coil_parameters, 3 * n_channels)

    offsets = Lvec[-1,:]

    Lvec = np.delete(Lvec, -1, axis=0)

    optimised_coil_settings, residuals, rank, s = np.linalg.lstsq(np.transpose(Lvec), -offsets, rcond=None)

    return optimised_coil_settings

def kalman_filter(coil_configurations, data_array, G=None):
    """
    Kalman filter to estimate ambient magnetic field from multiple measurements
    with different known coil configurations.
    """
    
    
    
    
    n_channels, _, n_samples = data_array.shape
    n_coil_parameters = coil_configurations.shape[1]
    state_dim = n_channels * 3  # Only magnetic field (Bx, By, Bz) at each sensor

    # Coil influence model (can be passed in or randomly initialized)
    if G is None:
        G = np.random.randn(state_dim, n_coil_parameters) * 0.01

    # Kalman filter matrices
    F = np.eye(state_dim)
    Q = np.eye(state_dim) * 1e-5
    R = np.eye(state_dim) * 1e-3
    x = np.zeros((state_dim, 1))
    P = np.eye(state_dim) * 1e-3

    def monitor_and_reset_P(P):
        eigenvalues = np.linalg.eigvals(P)
        if np.max(eigenvalues) > 1e3 or np.min(eigenvalues) < 1e-10:
            print("Resetting P due to numerical instability.")
            return np.eye(state_dim) * 1e-3
        return P

    def adapt_noise_covariance(P, R):
        innovation_variance = np.trace(P) / state_dim
        return np.eye(state_dim) * max(1e-4, min(1e-2, innovation_variance))

    for t in range(n_samples):
        measured_field = data_array[:, :, t].reshape(-1, 1)              # (n_channels*3, 1)
        coil_setting = coil_configurations[t].reshape(-1, 1)             # (n_coil_parameters, 1)

        # Measurement adjustment: remove known coil field effect
        adjusted_measurement = measured_field - G @ coil_setting

        # Prediction
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        H = np.eye(state_dim)  # Direct measurement of field
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ (adjusted_measurement - H @ x_pred)
        P = (np.eye(state_dim) - K @ H) @ P_pred

        # Adaptive noise tuning and stability check
        R = adapt_noise_covariance(P, R)
        P = monitor_and_reset_P(P)
        P += np.eye(state_dim) * 1e-6


        print(f"Step {t}: Estimated field norm = {np.linalg.norm(x):.4e}")
        print("this function runs, the question is whether it makes sense hehe. Think about passing previous g?")

    # After going through all data, compute coil settings to null the field
    optimal_coil_settings = -np.linalg.pinv(G) @ x  # shape: (n_coil_parameters, 1)

    return optimal_coil_settings.flatten(), x.flatten(), G



def kalman_filter_one_set_of_data(coil_settings, data, n_iterations = 1000):

    """
    Questions: Can we take recordings using mulitiple different coil settings into account?
    """
    def monitor_and_reset_P(P):
        """Monitor P and reset if it diverges or collapses."""
        eigenvalues = np.linalg.eigvals(P)
        if np.max(eigenvalues) > 1e3 or np.min(eigenvalues) < 1e-10:
            print("Resetting P due to numerical instability.")
            P = np.eye(state_dim) * 1e-3  # Reset to initial values
        return P

    def adapt_noise_covariance(P, R):
        """Adaptively update noise covariance based on observed variations."""
        innovation_variance = np.trace(P[:n_channels * 3, :n_channels * 3]) / (n_channels * 3)
        R = np.eye(n_channels * 3) * max(1e-4, min(1e-2, innovation_variance))
        
        return R
    
    def kalman_update(data, x, P, R):
        measured_field = np.array(data).reshape((n_channels * 3, 1))  # Ensure correct shape
    
        # Prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Measurement update
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x = x_pred + K @ (measured_field - H @ x_pred)
        P = (np.eye(state_dim) - K @ H) @ P_pred
        
        # Extract coil current recommendations
        coil_states = x[n_channels * 3: n_channels * 3 + n_coil_parameters].reshape((n_coil_parameters, 1))  # Ensure correct shape
        optimal_currents = -np.linalg.pinv(G[n_channels * 3:, :]) @ coil_states  # Adjust current based on state
        
        return optimal_currents, x, P
    
    n_channels = data.shape[0]
    n_coil_parameters = coil_settings.shape[-1]
    
    # State vector: Magnetic field at each sensor (Bx, By, Bz) + Coil currents
    state_dim = n_channels * 3 + n_coil_parameters  # 3 components (x, y, z) per sensor

    # Kalman filter matrices
    F = np.eye(state_dim)  # Assume no inherent field drift
    G = np.random.randn(state_dim, n_coil_parameters) * 0.01  # Coil influence model (to be calibrated)
    H = np.hstack((np.eye(n_channels * 3), G[:n_channels * 3, :]))  # Measurement model
    Q = np.eye(state_dim) * 1e-5  # Process noise covariance
    R = np.eye(n_channels * 3) * 1e-3  # Measurement noise covariance

    # Initialize state and covariance
    x = np.zeros((state_dim, 1))  # Initial field is assumed zero
    P = np.eye(state_dim) * 1e-3  # Initial uncertainty

    

    for iteration in range(n_iterations):
        R = adapt_noise_covariance(P, R)
        optimal_coil_currents, x, P = kalman_update(data, x, P, R)
            
        # Monitor and reset P if necessary
        P = monitor_and_reset_P(P)
            
        # Small covariance inflation to prevent collapse
        P += np.eye(state_dim) * 1e-6
            
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Recommended coil currents:", optimal_coil_currents.flatten())
    
    return optimal_coil_currents