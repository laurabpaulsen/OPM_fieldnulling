
"""
HEADLESS version of control GUI from quspin
"""

import socket
import threading
import struct
import select
import numpy as np
from collections import deque

import os
import logging

class OPMQuspinControl:
    def __init__(self, server_ip, history_seconds=1):

        # Initialize connection data
        self.connections = {
            8089: {"connected": False, "socket": None, "name": "Data Stream", "data": [], "total_samples": 0, "data_buffer": deque()},
            8090: {"connected": False, "socket": None, "page1": [], "page2": []},
            8091: {"connected": False, "socket": None, "name": "Text Display 2"},
            8092: {"connected": False, "socket": None, "name": "Command Channel"}
        }
        # Load commands
        #self.commands = self.load_commands("N1_Commands.txt")
        self.server_ip = server_ip
        self.history_seconds = history_seconds  # Default history length

        self.channel_names = [f"X{i+1}" for i in range(64)] + \
                     [f"Y{i+1}" for i in range(64)] + \
                     [f"Z{i+1}" for i in range(64)] + \
                     [f"AUX{i+1}" for i in range(64)]

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
            # check if any data is not zero

            # Store latest frame
            self.connections[port]["last_frame"] = data_array

            # Trim buffer if too long
            max_samples = self.history_seconds * cols_adjusted
            while self.connections[port]["total_samples"] > max_samples and len(self.connections[port]["data_buffer"]) > 0:
                oldest = self.connections[port]["data_buffer"].popleft()
                self.connections[port]["total_samples"] -= oldest.shape[1]

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

    def process_text_data(self, port, payload, rows, cols, frame_num, checksum, dual_page=False):
            """Process text data for display"""
            try:
                if dual_page:
                    # Handle dual page data (port 8090)
                    half_cols = cols // 2
                    data_array = np.frombuffer(payload, dtype=np.uint8).reshape(rows*2, half_cols)
                    data_3d = np.zeros((2, rows, half_cols), dtype=np.uint8)
                    data_3d[0] = data_array[:rows]
                    data_3d[1] = data_array[rows:]
                    
                    for page in range(2):
                        string_array = [self.uint8_array_to_string(row) for row in data_3d[page]]
                        self.connections[port][f"page{page+1}"] = string_array
                else:
                    # Handle single page data (port 8091)
                    data_array = [payload[i:i+cols] for i in range(0, len(payload), cols)]
                    string_array = [self.uint8_array_to_string(row) for row in data_array]

                    # Store latest frame
                    self.connections[port]["last_frame"] = string_array


            except Exception as e:
                self.log_message(f"Error processing text data: {e}")

    def on_history_change(self, event=None):
        """Handle history length change"""
        try:
            # Validate history length
            new_history = self.history_length.get()
            if new_history < 1:
                self.history_length.set(1)
            elif new_history > 20:
                self.history_length.set(20)
                
            # Apply the change if we have data
            if self.connections[8089]["data"]:
                data_length = len(self.connections[8089]["data"][0])
                max_samples = self.history_length.get() * data_length
                
                # Trim data if needed
                if self.connections[8089]["total_samples"] > max_samples:
                    self.connections[8089]["data"] = self.connections[8089]["data"][-max_samples:]
                    self.connections[8089]["total_samples"] = max_samples
                
                
            self.log_message(f"History length set to {self.history_length.get()} seconds")
        except ValueError:
            self.history_length.set(4)  # Default to 4 if invalid input
            self.log_message("Invalid history length value, reset to 4")


    def on_closing(self):
        """Handle application closing"""
        try:
            # Disconnect from all ports
            for port in self.connections:
                if self.connections[port]["connected"]:
                    self.disconnect(port)
        except Exception as e:
            print(f"Error during shutdown: {e}")

    def check_sensor_status(self, value="calibration"):

        delimiters = {
            "calibration": "CBS",
            "laser_locked": "LLS",
            "receiving_data": "CNT",
            "disabled": "DIS",
            "active_to_commands": "ACT"
            }

        delimiter = delimiters.get(value)

        status_all = []

        for row in self.connections[8090]["page2"]:    
            # check if there is a 1 or zero after the delimiter
            # add a space to the beginning of the row to avoid index errors when the delimiter is the first character
            row = " " + row 

            try:
                parts = row.split(delimiter)
                print(parts)
                status = parts[1][0]
                status_all.append(status)
            except IndexError:
                print(f"Error processing row: {row}")

        # Print all statuses at once
        print(f"{sum(1 for s in status_all if s == '1')} out of {len(status_all)} sensors {value}")