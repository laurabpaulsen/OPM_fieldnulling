# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:59:27 2021

@author: rasmus.zetter
"""

import threading, time, serial


class ComMonitorThread(threading.Thread):
    """ A thread for monitoring a COM port. The COM port is
        opened when the thread is started.

        _q:
            Queue for received data. Items in the queue are
            (data, timestamp) pairs, where data is a binary
            string representing the received data, and timestamp
            is the time elapsed from the thread's start (in
            seconds).

        status_q:
            Queue for received status/response messages for sent commands.

        led_q:
            Queue for LED indicator status.

        para_q:
            Queue for sensor parameter readout, such as temperature error.

        error_q:
            Queue for error messages. In particular, if the
            serial port fails to open for some reason, an error
            is placed into this queue.

        send_q:
            Queue for messages to send.

        monitor_msg_q:
            Queue for ComMonitorThread messages.

        port:
            The COM port to open. Must be recognized by the
            system.

        port_baud/stopbits/parity:
            Serial communication parameters

        port_timeout:
            The timeout used for reading the COM port. If this
            value is low, the thread will return data in finer
            grained chunks, with more accurate timestamps, but
            it will also consume more CPU.
    """
    def __init__(self,
                 device_name,
                 tx_q, rx_q,
                 monitor_msg_q,
                 port_num,
                 port_baud,
                 port_stopbits=serial.STOPBITS_ONE,
                 port_parity=serial.PARITY_NONE,
                 port_timeout=0.01,
                 verbose=False,
                 exc_callback=None):
        
        threading.Thread.__init__(self)
        
        
        self._callback = exc_callback

        self.serial_port = None
        self.device_name = device_name  
        
        self.alive    = threading.Event()
        self.alive.set()
        
        #Search for COM port belonging to ZEROFIELD, using STM uC name as ID
        if port_num == 'auto': 
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()

            for port, desc, hwid in sorted(ports):
                if verbose:        
                    print("{}: {} [{}]".format(port, desc, hwid))
                if "VID:PID=0483:374E" in hwid:
                    port_num = port
                    if verbose:
                        print('Found ZEROFIELD on port %s'%port)
                    break
            
            if port_num == 'auto':
                e = serial.SerialException("ERROR: Cannot find ZEROFIELD COM port. Please connect ZEROFIELD system and restart this program.") 
                
                if self._callback is None:
                    raise e
                else:
                    self._callback(str(e))
                    # print("ERROR: Cannot find ZEROFIELD COM port. Please connect ZEROFIELD system and restart this program.")
                
                self.alive.clear()
                return

        self.port_num = port_num
        
        self.serial_arg  = dict( port      = port_num,
                                 baudrate  = port_baud,
                                 stopbits  = port_stopbits,
                                 parity    = port_parity,
                                 timeout   = port_timeout)

        self.tx_q   = tx_q
        self.rx_q = rx_q
        self.monitor_msg_q=monitor_msg_q



        self.start_time = None
        
        print('Started ZEROFIELD communication on port %s'%self.port_num)


    def run(self):
        
        if self.alive.is_set():
            try:
                if self.serial_port:
                    self.serial_port.close()
                self.serial_port = serial.Serial(**self.serial_arg)
                self.monitor_msg_q.put("Port "+self.port_num+" open to "\
                                       +self.device_name)
            except serial.SerialException as e:
                
                if self._callback is None:
                    raise e
                else:
                    self._callback(str(e))
                # print("Failed to open port "+self.port_num\
                #                        +" to "+self.device_name\
                #                        +" --> Terminating ComMonitorThread")
                self.alive.clear()
                return

        # Restart the clock
        self.start_time = time.time()

        while self.alive.is_set():
            try:
                if self.serial_port.inWaiting():
                    line = self.serial_port.readline()
    
                    self.rx_q.put(line)
                    
                    if self.verbose:
                        print(line)
                        
            except serial.SerialException as e:
                
                if self._callback is None:
                    raise e
                else:
                    self._callback(str(e))
                # print("ERROR: "+str(e)+" --> Terminating ComMonitorThread")
                return
                
            
            

            if not self.tx_q.empty():
                send_data = self.tx_q.get_nowait()
                # DEBUG
                # print("send_q: "+str(send_data))
                # DEBUG
                self.serial_port.write(send_data)
                now = time.time()
                
                while(time.time() - now < 0.0005):
                    time.sleep(0)
                # print('sending')
            # while end
        # with end

        # clean up
        if self.serial_port:
            self.serial_port.close()

    def join(self, timeout=None):
        self.alive.clear()
        threading.Thread.join(self, timeout)
