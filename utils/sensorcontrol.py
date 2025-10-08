
"""
HEADLESS version of control GUI from quspin
"""

import socket
import threading
import struct
import select
import numpy as np

import os
import logging
import re

class OPMQuspinControl:
    def __init__(self, server_ip, max_samples=1000):

        # Initialize connection data
        self.connections = {
            8089: {"connected": False, "socket": None, "name": "Data Stream", "data": [], "total_samples": 0, "data_buffer": None},
            8090: {"connected": False, "socket": None, "page1": []},
            8091: {"connected": False, "socket": None, "name": "Text Display 2"},
            8092: {"connected": False, "socket": None, "name": "Command Channel"}
        }
        # Load commands
        #self.commands = self.load_commands("N1_Commands.txt")
        self.server_ip = server_ip

        self.channel_names = [f"X{i+1}" for i in range(64)] + \
                     [f"Y{i+1}" for i in range(64)] + \
                     [f"Z{i+1}" for i in range(64)] + \
                     [f"AUX{i+1}" for i in range(64)]
        self.max_samples = max_samples

        self.sensor_status = {}
        self.additional_status_info = [] 
        # additional status info desciphor 
        self.SENSOR_STATUS_INFO = ['Bz Fast lock','By Fast lock','Bx Fast lock',
                                   'Bz Slow lock','By Slow lock','Bx Slow lock',
                                   'Slow Closed Loop',
                                   'Bz MOD','By MOD','Bx MOD',
                                   'Calibration applied and in normal range','Calibration procedure active',
                                   'Fields from Field Zero is applied','Field Zero active',
                                   'Auto Start has completed','Auto Start procedure is running',
                                   'light check passed',
                                   'Triaxial mode'
                                    ] # 0-13 are not used and therefore removed from the message. 
        
        # self._status_lock = threading.Lock() # Attempt to grap snapshot of the sensor status

    def log_message(self, message):
        logging.info(message)

    def load_commands(self, filename):
        script_dir = os.path.dirname(os.path.realpath(__file__))  
        file_path = os.path.join(script_dir, filename)
        
        if not os.path.exists(file_path):
            self.log_message(f"Warning: {filename} not found. Command list will be empty.")
            return []
    
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    def connect_all_ports(self):
        """Connect to all ports at once"""
        for port in self.connections.keys():
            if not self.connections[port]["connected"]:
                self.connect(port)
        self.log_message("Attempted to connect to all ports")

    def disconnect_all_ports(self):
        """Disconnect from all ports at once"""
        for port in self.connections.keys():
            if self.connections[port]["connected"]:
                self.disconnect(port)
        self.log_message("Disconnected from all ports")

    def connect(self, port):
        """Connect to a specific port."""
        try:
            self.log_message(f"Connecting to port {port}...")

            socket_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket_obj.settimeout(5)  # 5 second timeout for connection
            socket_obj.connect((self.server_ip, port))  # <-- use plain attribute
            socket_obj.settimeout(None)  # Reset timeout for normal operation

            # Update connection status
            self.connections[port]["socket"] = socket_obj
            self.connections[port]["connected"] = True

            # Start receiving thread if this is a data/text port
            if port in [8089, 8090, 8091]:
                threading.Thread(target=self.receive_data, args=(port,), daemon=True).start()

            self.log_message(f"Connected to server on port {port}")

        except Exception as e:
            self.log_message(f"Connection failed on port {port}: {e}")

    def disconnect(self, port):
        """Disconnect from a specific port (headless version, no GUI)."""
        try:
            if self.connections[port]["socket"]:
                try:
                    self.connections[port]["socket"].close()
                except Exception as e:
                    self.log_message(f"Error closing socket on port {port}: {e}")
                finally:
                    self.connections[port]["socket"] = None

            self.connections[port]["connected"] = False
            self.log_message(f"Disconnected from server on port {port}")

        except Exception as e:
            self.log_message(f"Unexpected error disconnecting from port {port}: {e}")

    def send_message(self, message):
        """Send custom message from the entry field"""
        if not self.connections[8092]["connected"]:
            self.log_message("Not connected to command server on port 8092")
            return

        if message:
            self.send_data(message)
        else:
            self.log_message("Message is empty")

    def send_command(self, command):
        """Send selected predefined command"""
        if not self.connections[8092]["connected"]:
            self.log_message("Not connected to command server on port 8092")
            return
        if command:
            self.send_data(command)
        else:
            self.log_message("No command selected")

    def send_data(self, data):
        """Send data through port 8092"""
        try:
            size = len(data)
            size_bytes = struct.pack("!I", size)
            self.connections[8092]["socket"].sendall(size_bytes + data.encode())
            
        except Exception as e:
            self.log_message(f"Failed to send data: {e}")

    def receive_data(self, port):
        """Receive and process data from a specific port"""
        while self.connections[port]["connected"]:
            try:
                # Read header (20 bytes)
                header = self.read_exact(port, 20)
                if not header:
                    break

                # Parse header
                magic, frame_num, payload_size, rows, cols = struct.unpack("<4sIIII", header)
                
                # Validate magic number
                if magic != b'KCLB':
                    self.log_message(f"Invalid start magic number on port {port}")
                    continue

                # Read payload
                payload = self.read_exact(port, payload_size)
                if not payload:
                    continue

                # Read footer (12 bytes)
                footer = self.read_exact(port, 12)
                if not footer:
                    continue

                # Parse footer
                end_magic, checksum, final_magic = struct.unpack("<4sI4s", footer)

                # Validate footer magic numbers
                if end_magic != b'DNEB' or final_magic != b'KCLB':
                    self.log_message(f"Invalid end or final magic number on port {port}")
                    continue

                # Process data based on port
                if port == 8089:
                    self.process_graph_data(port, payload, rows, cols, frame_num)
                elif port == 8090:
                    self.process_text_data(port, payload, rows, cols, frame_num, checksum, dual_page=True)
                elif port == 8091:
                    self.process_text_data(port, payload, rows, cols, frame_num, checksum)

            except Exception as e:
                self.log_message(f"Error receiving data on port {port}: {e}")
                break

        self.log_message(f"Stopped receiving data from port {port}")
    
    def process_graph_data(self, port, payload, rows, cols, frame_num):
        try:
            cols_adjusted = round(cols * 0.042667)
            if cols_adjusted <= 0:
                self.log_message(f"Skipping frame {frame_num}: cols_adjusted={cols_adjusted}")
                return

            data_array = np.frombuffer(payload, dtype=np.float32).reshape(rows * 4, cols_adjusted)
            # check if any data is not zero | this isnot actually being checked right?

            # Store latest frame
            self.connections[port]["last_frame"] = data_array # shape is channels, time

            # Append to buffer
            if self.connections[port]["data_buffer"] is None:
                self.connections[port]["data_buffer"] = data_array
            else:
                self.connections[port]["data_buffer"] = np.concatenate((self.connections[port]["data_buffer"], data_array), axis=1)

            self.connections[port]["total_samples"] += data_array.shape[-1]

            # Trim buffer if too long
            if self.connections[port]["total_samples"] > self.max_samples and self.connections[port]["data_buffer"] is not None:
                excess = self.connections[port]["total_samples"] - self.max_samples
                self.connections[port]["data_buffer"] = self.connections[port]["data_buffer"][:, excess:]
                self.connections[port]["total_samples"] = self.max_samples

        except Exception as e:
            self.log_message(f"Error processing graph data: {e}")

    def read_exact(self, port, byte_count):
        """Read exactly byte_count bytes from the socket"""
        data = b''
        while len(data) < byte_count:
            # Validate socket and connection
            if not self.connections[port]["socket"] or not self.connections[port]["connected"]:
                return None

            # Wait for data with timeout
            ready = select.select([self.connections[port]["socket"]], [], [], 5.0)
            if ready[0]:
                chunk = self.connections[port]["socket"].recv(byte_count - len(data))
                if not chunk:  # Connection closed
                    self.disconnect(port)  # <-- just call disconnect directly
                    return None
                data += chunk
            else:
                self.log_message(f"Timeout waiting for data on port {port}")
                return None

        return data

    def uint8_array_to_string(self, uint8_array):
        """Convert uint8 array to string, filtering non-displayable characters"""
        return ''.join(chr(b) if (32 <= b <= 126) or b == 20 else '' for b in uint8_array)

    def update_sensor_status(self, status_data):
        """Update sensor status information"""
        string_rows = [self.uint8_array_to_string(row) for row in status_data]

        for i, row in enumerate(string_rows):
            #print(row)
            values = re.findall(r'([A-Z]{3})(-?\d+(?:\.\d+)?)', row)

            self.sensor_status[i] = {prefix: number for prefix, number in values} # This variable is not updated in our data collected on the Sept. 5th 2025 -> we need to make sure this happens
    
    def uint32_t_to_bool(self):
        # status_message = [str(format(self.sensor_status[key]['STS'], "032b"))[13:] for key in self.sensor_status]
        self.additional_status_info = []
        
        for i in self.sensor_status:
            status_message = str(format(int(self.sensor_status[i]["STS"]), "032b"))[13:]
            self.additional_status_info.append(status_message)
 
    def process_text_data(self, port, payload, rows, cols, frame_num, checksum, dual_page=False):
            """Process text data for display"""
            try:
                if dual_page:
                    # Handle dual page data (port 8090)
                    half_cols = cols // 2
                    data_array = np.frombuffer(payload, dtype=np.uint8).reshape(rows*2, half_cols)

                    self.update_sensor_status(data_array[rows:])
                    self.uint32_t_to_bool()

                    string_array = [self.uint8_array_to_string(row) for row in data_array[:rows]]
                    self.connections[port]["page1"] = string_array

                else:
                    # Handle single page data (port 8091)
                    data_array = [payload[i:i+cols] for i in range(0, len(payload), cols)]
                    string_array = [self.uint8_array_to_string(row) for row in data_array]

                    # Store latest frame
                    self.connections[port]["last_frame"] = string_array


            except Exception as e:
                self.log_message(f"Error processing text data: {e}")

    def on_closing(self):
        """Handle application closing"""
        try:
            # Disconnect from all ports
            for port in self.connections:
                if self.connections[port]["connected"]:
                    self.disconnect(port)
        except Exception as e:
            print(f"Error during shutdown: {e}")

    def get_fields(self):
        """gets the fields averaged over self.max_samples"""
        if not self.connections[8091]["data_buffer"]:
            return None

        data = self.connections[8091]["data_buffer"]
        return data.mean(axis=1)
    
    def wait_im_not_done(self,command):
        # from progressbar import ProgressBar
        from time import sleep
        self.send_command(command)
        try:
            match command:
                case 'Sensor|Reboot':
                    info_idx = 16
                case 'Sensor|Auto Start':
                    info_idx = 14
                case _: # Default/fail safe option 
                    self.log_message('Not valid Command registered!')
        except:
            self.log_message('Not sure how we ended up here?')

        self.log_message(f'Chekcing: {self.SENSOR_STATUS_INFO[info_idx]}')
        info_check = []
        # with ProgressBar(max_value=10) as bar:
        for _ in range(3*60*2): # limit loop to 2 minuts 
            info_check = [int(key[info_idx]) for key in self.additional_status_info]
            sleep(0.333)
            # bar.update(sum(info_check))
            print(sum(info_check))
            if all(info_check):
                break
        # for _ in range(30):

