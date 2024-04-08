import numpy as np
from rx import subject
import EEGDataFilterUtil
import EEGDataBean

class EEGDataTransformHelper:
    channel_count = 32    # 通道数
    data_count = 1        # 每个通道的数据点数量

    is_50Hz_comb = False
    is_40Hz_low_pass = False
    is_70Hz_low_pass = False

    def __init__(self):
        self.logger = None
        self.TAG = "DataTransformHelper"
        self.ADC = 8388608  # ADC
        self.BASE_VOLTAGE = 4.5  # 基准电压
        self.EEG1_CIR_GAIN = 12  # 脑电1默认有12倍增益
        self.EEG2_CIR_GAIN = 12  # 脑电2默认有12倍增益
        self.vpp_value = [0.0] * 64
        self.buffer_size = 2000
        self.vpp_buffer = np.zeros((64, self.buffer_size))
        self.offset_vpp = [0] * 64
        self.vpp_ok = subject.Subject()

    def parse_eeg_data_no_flag(self, data, comb, low_pass):
        eeg_list = []
        for i in range(self.data_count):
            eeg_data = EEGDataBean.EEGChannelBean(True, True)
            for j in range(self.channel_count):
                high = data[i * (self.channel_count * 3) + j * 3 + 2]
                middle = data[i * (self.channel_count * 3) + j * 3 + 1]
                low = data[i * (self.channel_count * 3) + j * 3]
                # value = self.put_byte_3(low, middle, high)
                value = (low << 16) | (middle << 8) | high
                raw_value = self.eeg_data_transform(value)
                if comb:
                    raw_value = EEGDataFilterUtil.comb_filter(raw_value, j)
                if low_pass:
                    raw_value = EEGDataFilterUtil.low_pass_40Hz_filter(raw_value, j)
                self.vpp_transform(raw_value, j)
                eeg_data.eeg_value[j] = raw_value
            eeg_list.append(eeg_data)
        return EEGDataBean(eeg_list)

    def eeg_data_transform(self, i):
        if i > self.ADC:
            i = (i - 2 * self.ADC)
        return (i * self.BASE_VOLTAGE * 1000 / self.ADC / self.EEG1_CIR_GAIN)

    def vpp_transform(self, data, i):
        self.vpp_buffer[i][self.offset_vpp[i]] = data
        self.offset_vpp[i] += 1

        if self.offset_vpp[i] == self.buffer_size:
            self.offset_vpp[i] = 0
            v_max = np.amax(self.vpp_buffer[i])
            v_min = np.amin(self.vpp_buffer[i])
            self.vpp_value[i] = v_max - v_min
            self.vpp_ok.on_next(self.vpp_value)
