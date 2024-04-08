import array

# 脑电信号数据包
class EEGPacketData:
    def __init__(self, raw_data: str, frame_header: bytes, info: bytes):
        self.raw_data = raw_data  # 原始数据的十六进制格式
        self.frame_header = frame_header
        self.info = info

    def get_raw_data(self) -> str:
        return self.raw_data

    def get_frame_header(self) -> bytes:
        return self.frame_header

    def get_info(self) -> bytes:
        return self.info

    def __str__(self) -> str:
        return f"EEGPacketData(frame_header={array.array('B', self.frame_header)}, info={array.array('B', self.info)})"