from typing import List, Set, Dict, Tuple, NoReturn 

class EEGDataBean:
    def __init__(self, eeg_list_channel:List[EEGChannelBean]):
        self.eeg_list_channel = eeg_list_channel
    
    def getEegListChannel(self):
        return self.eeg_list_channel

class EEGChannelBean:
    def __init__(self, is_marked=True, is_drop=True) -> NoReturn :
        self.eeg_value = [0.0] * 64  # 64个通道的数据
        self.is_marked = is_marked
        self.is_drop = is_drop