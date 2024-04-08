import EEGDataBean

class EEGInfoDataBean:
    def __init__(self, eeg_data: EEGDataBean):
        self.eeg_data : EEGDataBean = eeg_data
        self.checksum = None  # 初始化为 None

    def get_eeg_data(self):
        return self.eeg_data

    def get_checksum(self):
        return self.checksum

    def __str__(self):
        return f"EEGInfoDataBean{{eeg_data={self.eeg_data}}}"
