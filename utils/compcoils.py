import sys 
import time
import queue
import numpy as np
from bitarray import util
sys.path.append('../..')
sys.path.append('coilAPI')
from utils.com_monitor import ComMonitorThread

#from .optim import nonneg_residual_lsq_algorithm

class CompFieldControl:
    def __init__(self, min_voltage = -10, max_voltage = 10):#, optimisation_algorithm = nonneg_residual_lsq_algorithm):
        self.tx_q = queue.Queue(maxsize=10000)
        self.rx_q = queue.Queue(maxsize=10000)
        self.monitor_msg_q = queue.Queue(maxsize=100)
        self.ser_monitor = ComMonitorThread(
            'ZEROFIELD',
            self.tx_q,
            self.rx_q,
            self.monitor_msg_q,
            'auto',
            921600,
            verbose=None,
            exc_callback = None
        )
    
        self.ch_names = ['Y','Z','X','dBy/dy','dBz/dy','dBz/dz','dBz/dx','dBy/dx']
        self.coil_dBdI = np.array([0.4570,0.6372,0.9090,2.6369,2.6271,1.4110,4.8184,2.4500]) 
        self.scales = np.array([1e-6, 1e-6, 1e-6, 1e-6/100, 1e-6/100, 1e-6/100, 1e-6/100, 1e-6/100]) 
        self.coil_dBdI = self.coil_dBdI * self.scales #in T/A(/cm) 
        self.coil_R = np.array([19.52,13.93,12.16,18.48,16.67,14.18,10.39,8.51]) # Measured values, in Ohms
        self.coil_dBdV = self.coil_dBdI / self.coil_R
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        #self.optimisation_algorithm = optimisation_algorithm

        time.sleep(3)
        self.ser_monitor.start()

    def setOffset(self, ch, field, verbose=True, delay=0.1):

        value = field / self.coil_dBdV[ch] * 1e-9
        
        if verbose:
            print("%s (CH%d): offset magnetic field %.6f nT"%(self.ch_names[ch], ch+1, field))
        
        cmdbyte = util.int2ba(ch, length=8).tobytes()
        
        uintbytes= util.int2ba(int((2**19-1) * -value/10 + (2**19-1)), length=32).tobytes()
        
        self.tx_q.put_nowait(cmdbyte + uintbytes[::-1])
        time.sleep(delay)

    def print_rx(self):
    
        if not self.rx_q.empty():
            rx_data = self.rx_q.get_nowait()
        
            print("Rx:")
            print(rx_data)

    #def optimise_coil_settings(self, coil_values, data_array, kwargs={}):
    #    new_coil_values = self.optimisation_algorithm(coil_values, data_array, **kwargs)
        
    #    return np.round(new_coil_values, 2)


    def set_coil_values(self, values):
        self.setOffset(0, values[0]) 
        self.setOffset(1, values[1]) 
        self.setOffset(2, values[2]) 
        self.setOffset(3, values[3]) 
        self.setOffset(4, values[4]) 
        self.setOffset(5, values[5]) 
        self.setOffset(6, values[6]) 
        self.setOffset(7, values[7]) 
        time.sleep(2) # how important was this delay again????
