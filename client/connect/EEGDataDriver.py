import json
import time
import csv
from typing import List
import connect.EEGDataValidJudgeHelper as EEGDataValidJudgeHelper
from connect.EEGPacketDataParser import parse_packet,parse_info_data
import pdb

class EEGDataDriver:
    def __init__(self):
        self.SIGNAL_SIZE = 2000  # SINGNAL_SIZE = 
        self.DOWNSAMPLE_SIZE = 250
        self.maxDataChannel = 32
        self.dataChannel = 32
        self.is50HzComb = True
        self.is40HzLowPass = True
        self.eegData = [[0.0] * self.SIGNAL_SIZE for _ in range(self.maxDataChannel)]
        self.eegDataBuffer = [[0.0] * self.DOWNSAMPLE_SIZE for _ in range(self.maxDataChannel)]
        self.timestamp = 0
        self.dataIndex = 0
        self.mentalStateInferenceCount = 0
        self.mentalStateInferenceLimit = 3

    def parse_sampling(self, input: bytes):
        # raw_str = self.bytes_to_hex_string(input)
        # Assuming EEGPacketData and EEGInfoData classes are defined elsewhere
        eeg_packet_data = parse_packet(input)
        
        if EEGDataValidJudgeHelper.is_valid_of_frame_header(eeg_packet_data.frame_header):
            try:
                eeg_info_data = parse_info_data(eeg_packet_data.info, self.is50HzComb, self.is40HzLowPass)
                for eeg_channel in eeg_info_data.eeg_data.eeg_list_channel:
                    for i in range(self.dataChannel):
                        self.eegData[i][self.dataIndex] = eeg_channel.eeg_value[i]
                    self.dataIndex += 1
                    if self.dataIndex == self.SIGNAL_SIZE:
                        for i in range(125):
                            for j in range(self.dataChannel):
                                self.eegDataBuffer[j][i] = self.eegData[j][i * 8]       # eegDataBuffer[j][i]为数据处理完得到的脑电数据,从eegData里面抽样的，
                        self.timestamp = int(time.time() * 1000)                        # 这里是每隔8个数据抽样一次
                        self.dataIndex = 0
            except Exception as e:
                # logger.error("EEG parse error: " + str(e))
                print("EEG parse error: " + str(e))

    def get_eeg_data(self):
        return self.eegDataBuffer

    def set_eeg_data(self, eeg_data):
        for i in range(len(eeg_data)):
            for j in range(len(eeg_data[0])):
                self.eegDataBuffer[i][j] = eeg_data[i][j]
        self.timestamp = int(time.time() * 1000)
        # SpringBeanUtil.get_bean(MainServer).mental_model_inference()

    def get_eeg_data_timestamp(self):
        return self.timestamp if self.timestamp != 0 else int(time.time() * 1000)

    def has_eeg_data(self):
        return self.timestamp != 0

    def clear_eeg_data_buffer(self):
        for i in range(self.dataChannel):
            for j in range(self.DOWNSAMPLE_SIZE):
                self.eegDataBuffer[i][j] = 0.0

    def set_data_channel(self, data_channel):
        self.dataChannel = data_channel

    @staticmethod
    def get_json_string_of_eeg_data(data: List[List[float]]) -> str:
        eeg_data_map = {}
        for i in range(len(data)):
            eeg_data_map[str(i + 1)] = ' '.join(map(str, data[i]))
        return json.dumps(eeg_data_map)

    @staticmethod
    def get_json_string_of_partial_eeg_data(data: List[List[float]], start_index: int, length: int) -> str:
        eeg_data_map = {}
        for i in range(len(data)):
            eeg_data_map[str(i + 1)] = ' '.join(map(str, data[i][start_index:start_index + length]))
        return json.dumps(eeg_data_map)

    @staticmethod
    def get_eeg_sample_data_from_csv_file(csv_file_path: str) -> List[List[float]]:
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            eeg_data = []
            for row in reader:
                eeg_data.append([float(value) for value in row[2:]])
        return eeg_data
