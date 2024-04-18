# 改动采样率的时候，记得也要调整EEGDataDriver的SIGNAL_SIZE，
# 具体看self.eegDataBuffer[j][i] = self.eegData[j][i * 4] 
# i*4(i是DOWNSAMPLE_SIZE的索引) = SIGNAL_SIZE
# 不然会越界

DOWNSAMPLE_SIZE = 500
SIGNAL_SIZE = DOWNSAMPLE_SIZE * 4
CHANEL_NUM = 32

LABEL_MAP = {'up':0, 'down':1, 'left':2, 'right':3}