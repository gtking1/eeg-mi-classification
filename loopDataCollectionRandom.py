import argparse
import logging
import time
from collections import deque
from playsound import playsound

import brainflow
# import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
# from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import pandas as pd
import datetime

from tkinter import *
from tkinter.ttk import *

import random

# class Shape:
#     def __init__(self, master=None):
#         self.master = master
        
#         self.create()
    
#     def create(self):
        
#         self.canvas = Canvas(self.master)

#         self.canvas.config(background='black')

#         # self.canvas.create_oval(10, 10, 80, 80, 
#         #                     outline="black", fill="white",
#         #                     width=2)
        
#         # 1470, 956

#         points = [735, 100, 368, 856, 1102, 856]
        
#         self.canvas.create_polygon(points, outline="red",
#                             fill="red", width=2)

#         self.canvas.pack(fill=BOTH, expand=1)

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        print(len(self.exg_channels))
        #print(self.exg_channels)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        print(self.sampling_rate)
        self.samples_per_segment = 625 # 1250
        self.open_segments = 0
        self.closed_segments = 0
        self.session_time = datetime.datetime.now().strftime("%m_%d_%H_%M")

        self.eyes_open = random.randint(0, 1)
        self.num_each_seg = 48 # 12

        #_ = self.board_shim.get_board_data()

        win = Tk()
        win.done = False
        win.protocol("WM_DELETE_WINDOW", lambda: setattr(win, 'done', True))
        win.title("Main Window")
        if self.eyes_open == 0:
            win.configure(bg='red')
        else:
            win.configure(bg='blue')
        win.attributes('-fullscreen',True)
        # shape = Shape(win)

        while ((self.open_segments == self.closed_segments) and self.open_segments == self.num_each_seg) == False and win.done == False:
            win.update()
            if self.board_shim.get_board_data_count() >= self.samples_per_segment:
                data = self.board_shim.get_board_data()
                print(data[0], datetime.datetime.now())
                print()
                labels = np.full((1, data.shape[1]), self.eyes_open)
                data = np.concatenate((data, labels), axis=0)
                DataFilter.write_file(data, f'{self.session_time}.csv', 'a')  # use 'a' for append mode
                self.eyes_open = random.randint(0, 1)
                if self.eyes_open == 0:
                    playsound('./beep.mp3', block=False)
                    print("Rest")
                    win.configure(bg='black')
                    win.update()
                    time.sleep(5)
                    print(self.board_shim.get_board_data_count())
                    data = self.board_shim.get_board_data()
                    print(self.board_shim.get_board_data_count())

                    playsound('./beep.mp3', block=False)
                    print("Showing red screen")
                    win.configure(bg='red')
                    self.closed_segments += 1
                elif self.eyes_open == 1:
                    playsound('./beep.mp3', block=False)
                    print("Rest")
                    win.configure(bg='black')
                    win.update()
                    time.sleep(5)
                    print(self.board_shim.get_board_data_count())
                    data = self.board_shim.get_board_data()
                    print(self.board_shim.get_board_data_count())

                    playsound('./beep.mp3', block=False)
                    print("Showing blue screen")
                    win.configure(bg='blue')
                    self.open_segments += 1
                # elif self.eyes_open == 1 or self.eyes_open == 3:
                #     playsound('./beep.mp3', block=False)
                #     print("Close eyes")

            if ((self.open_segments == self.closed_segments) and self.open_segments == self.num_each_seg) == True:
                print("Ending data collection")
                time.sleep(3)

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    useRealBoard = True
    
    if useRealBoard == False:
        # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
        parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                            default=0)
        parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
        parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                            default=0)
        parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
        parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
        parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
        parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
        parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
        parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                            required=False, default=BoardIds.SYNTHETIC_BOARD)
        parser.add_argument('--file', type=str, help='file', required=False, default='')
        parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                            required=False, default=BoardIds.NO_BOARD)

    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()

    if useRealBoard:
        params.serial_port = "/dev/cu.usbserial-DP04VZS3"
        board_id = BoardIds.CYTON_DAISY_BOARD
        board_shim = BoardShim(board_id, params)
    else:
        params.ip_port = args.ip_port
        params.serial_port = args.serial_port
        params.mac_address = args.mac_address
        params.other_info = args.other_info
        params.serial_number = args.serial_number
        params.ip_address = args.ip_address
        params.ip_protocol = args.ip_protocol
        params.timeout = args.timeout
        params.file = args.file
        params.master_board = args.master_board

        board_shim = BoardShim(args.board_id, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params) # 450000 is number of samples in ring buffer
        print("In 5 seconds, prepare to clench right fist for 10 seconds")
        time.sleep(5)
        playsound('./beep.mp3', block=False)
        print("Starting recording")
        board_shim.get_board_data()
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()

if __name__ == '__main__':
    main()