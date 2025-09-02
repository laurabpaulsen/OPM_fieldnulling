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

    use_all_data_for_updating_fields = True # added functionality - needs to be tested

    if use_all_data_for_updating_fields:
        all_coil_configurations = []
        all_data_arrays = []


    rescale_steps = np.array([1, 1, 1, 0.15, 0.15, 0.15, 0.15, 0.15])
    #coil_parameters = starting_point_coil_vals(output_path, which = args.start_coil_vals)


    try:
        OPM_control = OPMQuspinControl(server_ip = "192.168.0.10")
        comp_coils = CompFieldControl()
        start_time = time.time()

        try:
            print("connecting the ports!")
            # connect to port 8089
            OPM_control.connect(8089)
            # Collect and print a few frames
            n_frames_to_print = 5
            
            for _ in range(5):
                frame = OPM_control.connections[8089].get("last_frame")
                if frame is not None:
                    print(f"Latest frame: with shape {frame.shape}")
                    print(frame)

                #if 8089 in OPM_control.connections and "last_frame" in OPM_control.connections[8089]:
                #    frame = OPM_control.connections[8089]["last_frame"]
                #    if frame is not None:
                #        print("Latest frame from Data Stream port 8089:")
                #        print(frame)
                # Sleep a bit to allow next frame to arrive
                time.sleep(0.1)

        except Exception as e:
            logging.error(f"Error during OPM control operations: {str(e)}")

    except ConnectionError as e:
        logging.error(f"Failed to connect: {str(e)}")

