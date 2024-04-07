import numpy as np
from rx import subject

class EEGDataBean:
    def __init__(self, eeg_list_channel):
        self.eeg_list_channel = eeg_list_channel

    def get_eeg_list_channel(self):
        return self.eeg_list_channel

class EEGChannelBean:
    def __init__(self, is_marked, is_drop):
        self.eeg_value = np.zeros(64)
        self.is_marked = is_marked
        self.is_drop = is_drop

class EEGDataTransformHelper:
    channel_count = 32  # 通道数
    data_count = 1  # 每个通道的数据点数量

    is_50_hz_comb = False
    is_40_hz_low_pass = False
    is_70_hz_low_pass = False

    @staticmethod
    def parse_eeg_data_no_flag(data, comb, low_pass):
        eeg_list = []
        for i in range(EEGDataTransformHelper.data_count):
            eeg_data = EEGChannelBean(True, True)
            for j in range(EEGDataTransformHelper.channel_count):
                high = data[i * (EEGDataTransformHelper.channel_count * 3) + j * 3 + 2]
                middle = data[i * (EEGDataTransformHelper.channel_count * 3) + j * 3 + 1]
                low = data[i * (EEGDataTransformHelper.channel_count * 3) + j * 3]
                value = (low << 16) | (middle << 8) | high
                raw_value = EEGDataTransformHelper.eeg_data_transform(value)
                if comb:
                    raw_value = EEGDataTransformHelper.comb_filter(raw_value, j)
                if low_pass:
                    raw_value = EEGDataTransformHelper.low_pass_40_hz_filter(raw_value, j)
                vpp_ok.onNext(vpp_value)
                eeg_data.eeg_value[j] = raw_value
            eeg_list.append(eeg_data)
        return EEGDataBean(eeg_list)

    @staticmethod
    def eeg_data_transform(i):
        adc = 8388608
        base_voltage = 4.5
        eeg1_cir_gain = 12
        if i > adc:
            i = (i - 2 * adc)
        return (i * base_voltage * 1000) / (adc * eeg1_cir_gain)

    @staticmethod
    def comb_filter(data, i):
        return data  # Placeholder implementation

    @staticmethod
    def low_pass_40_hz_filter(data, i):
        return data  # Placeholder implementation

vpp_value = np.zeros(64)
BUFFER_SIZE = 2000
vpp_buffer = np.zeros((64, BUFFER_SIZE))
offset_vpp = np.zeros(64, dtype=int)
vpp_ok = subject.Subject()

def vpp_transform(data, i):
    global vpp_value, vpp_buffer, offset_vpp
    vpp_buffer[i][offset_vpp[i]] = data

    if offset_vpp[i] == BUFFER_SIZE:
        offset_vpp[i] = 0
        v_max = np.max(vpp_buffer[i])
        v_min = np.min(vpp_buffer[i])
        vpp_value[i] = v_max - v_min
        vpp_ok.on_next(vpp_value)
