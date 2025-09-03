"""
Slightly modified from the work done by Jacob Væver Andersen! Shoutout!
"""

import time
import logging

from pathlib import Path
import argparse as ap

import numpy as np
from utils.compcoils import CompFieldControl
from utils.sensorcontrol import OPMQuspinControl
    
#from utils.data_handling import starting_point_coil_vals, save_array_to_txt

#from fieldline_api.fieldline_service import FieldLineService

#from nulling_my_coils import create_matrix_of_coil_vals, remove_failed_chs, collect_data_array

stream_handler = logging.StreamHandler()
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(threadName)s(%(process)d) %(message)s [%(filename)s:%(lineno)d]',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    handlers=[stream_handler]
)


def create_matrix_of_coil_vals(start_vec, rescale_step):
    """
    Generates a matrix of coil values.

    This function constructs a lower triangular matrix with values below the diagonal, 
    scales it using a given `rescale_step`, adds an initial `start_vec`, and appends 
    a column of ones.

    Parameters:
    -----------
    start_vec : array-like
        A 1D array representing the starting values.
    
    rescale_step : array-like
        A 1D array with the same length as `start_vec`, specifying the scaling factors 
        for each column in the lower triangular matrix.

    Returns:
    --------
    coil_values : numpy.ndarray
        A (len(start_vec) + 1, len(start_vec) + 1) matrix where:
        - The first `len(start_vec)` columns define the adjusted values based on 
          `start_vec` and `rescale_step`.
        - The last column is filled with ones.
    """
    n_rows = len(start_vec) + 1
    n_cols = len(start_vec)

    # Create a lower triangular matrix with ones below the diagonal
    lower_tri_matrix = np.tril(np.ones((n_rows, n_cols)), -1)

    # Scale the lower triangular matrix using rescale_step
    scaled_matrix = lower_tri_matrix * rescale_step

    # Add start_vec to each row
    adjusted_matrix = scaled_matrix + start_vec

    # Append a column of ones
    coil_values = np.concatenate((adjusted_matrix, np.ones((n_rows, 1))), axis=1)

    return coil_values


def collect_data_array(start_vec, rescale_step, compcoils_control:CompFieldControl, OPM_control:OPMQuspinControl, active_sensors):
    """
    This function assumes that the coil values are already set to the start vec. 
    Then the coil values are changed one at a time according to the step size and records the fields (x, y, z) across the sensors. 
    """
    # prepare a matrix with the coil values at each iteration when we change the coil parameters one at a time. The first row is the start vec.
    coil_values = create_matrix_of_coil_vals(start_vec, rescale_step)

    # preallocate memory for data collected from changing the coil parameters
    data = np.empty((len(active_sensors), 3, len(start_vec)+1), float) 


    for j, coil_values_tmp in enumerate(coil_values):
        # add 1 stepsize to each of the values in start_vec for each component of the field
        if j != 0: # as we have already set the field to the starting value we will not change it here  
            compcoils_control.setOffset(j-1, coil_values_tmp[j-1])
            
        #opm_main.fine_zero_sensors()
        # get the data in the databuffer
        data_tmp = OPM_control.connections[8089].get("data_buffer")
        data_tmp = data_tmp[:64*3] # ignore AUX

        # so we have sensor and channel dimensions
        n, n_samples = data_tmp.shape[0], data_tmp.shape[1]
        data_tmp = data_tmp.reshape(3, int(n/3), n_samples)

        # only get active sensors
        data_active = data_tmp[:, active_sensors, :]

        # average over time dimension
        data_active_mean = data_active.mean(axis=2)
        data[:,:,j] = data_active_mean


    return coil_values, data




def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--nulling_type', type=str, default='full', help='Whether to complete the "full" nulling procedure or "finetune"')
    parser.add_argument('--n_channels', type=int, default=16, help='The expected number of OPM channels')
    parser.add_argument('--start_coil_vals', type=str, default='reliable_initial_guess', help='Which values to start the compensation coils at')

    args = parser.parse_args()
    
    return args



if __name__ == "__main__":

    path = Path(__file__).parent

    output_path = path / "previous_nulling_params"

    args = parse_args()

    if args.nulling_type == "full":
        n_iterations = 3

    elif args.nulling_type == "finetune":
        n_iterations = 1


    rescale_steps = np.array([1, 1, 1, 0.15, 0.15, 0.15, 0.15, 0.15])
    #coil_parameters = starting_point_coil_vals(output_path, which = args.start_coil_vals)



    compcoils = CompFieldControl()
    #compcoils.set_coil_values([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


    OPM_control = OPMQuspinControl(server_ip = "192.168.0.10")
    #comp_coils = CompFieldControl()
    start_time = time.time()

    OPM_control.connect_all_ports()

    OPM_control.send_command("Sensor|Ortho & Calibrate") 
    time.sleep(2)

    # get the keys where sensor_status[key]["LLS"] is "1"
    active_sensors = [key for key in OPM_control.sensor_status if OPM_control.sensor_status[key]["LLS"] == "1"]
    

    coil_vals, collected_data_array = collect_data_array(
        np.array([0, 0, 0, 0, 0, 0, 0, 0]), rescale_steps, compcoils, OPM_control, active_sensors)

    """
    n_frames_to_print = 5

    for _ in range(n_frames_to_print):
        frame = OPM_control.connections[8089].get("last_frame")
        if frame is not None:
            print(f"Latest frame: with shape {frame.shape}")
            print(frame[20])

                #if 8089 in OPM_control.connections and "last_frame" in OPM_control.connections[8089]:
                #    frame = OPM_control.connections[8089]["last_frame"]
                #    if frame is not None:
                #        print("Latest frame from Data Stream port 8089:")
                #        print(frame)
                # Sleep a bit to allow next frame to arrive
            time.sleep(0.1)
    """
    OPM_control.disconnect_all_ports()
    compcoils.close()

