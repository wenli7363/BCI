import numpy as np
from rx.subject import Subject
from EEGDataFilterUtil import combFilter, lowPass40HzFilter
from EEGDataBean import EEGDataBean, EEGChannelBean

channelCount = 32  # 通道数
dataCount = 1  # 每个通道的数据点数量

is50HzComb = False
is40HzLowPass = False
is70HzLowPass = False

TAG = "DataTransformHelper"

ADC = 8388608  # ADC
BASE_VOLTAGE = 4.5  # 基准电压
EEG1_CIR_GAIN = 12  # 脑电1默认有12倍增益
EEG2_CIR_GAIN = 12  # 脑电2默认有12倍增益

BUFFER_SIZE = 2000
vppBuffer = np.zeros((64, BUFFER_SIZE))
offsetVpp = np.zeros(64, dtype=int)
vppOk = Subject()

vppValue = np.zeros(64)

def parseEEGDataNoFlag(data, comb, lowPass):
    eegList = []
    for _ in range(dataCount):
        eegData = EEGChannelBean(True, True)
        for j in range(channelCount):
            high = data[_ * (channelCount * 3) + j * 3 + 2]
            middle = data[_ * (channelCount * 3) + j * 3 + 1]
            low = data[_ * (channelCount * 3) + j * 3]
            value = (low << 16) | (middle << 8) | high
            rawValue = eegDataTransform(value)
            if comb:
                rawValue = combFilter(rawValue, j)
            if lowPass:
                rawValue = lowPass40HzFilter(rawValue, j)
            vppTransform(rawValue, j)
            eegData.eeg_value[j] = rawValue
        eegList.append(eegData)
    return EEGDataBean(eegList)

def eegDataTransform(i):
    if i > ADC:
        i -= 2 * ADC
    return i * BASE_VOLTAGE * 1000 / ADC / EEG1_CIR_GAIN

def vppTransform(data, i):
    global offsetVpp, vppBuffer, vppValue
    vppBuffer[i][offsetVpp[i]] = data
    offsetVpp[i] += 1
    if offsetVpp[i] == BUFFER_SIZE:
        offsetVpp[i] = 0
        vMax = np.max(vppBuffer[i])
        vMin = np.min(vppBuffer[i])
        vppValue[i] = vMax - vMin
        vppOk.on_next(vppValue)
