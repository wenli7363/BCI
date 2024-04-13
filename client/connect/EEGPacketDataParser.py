from connect.EEGPacketDataBean import EEGPacketData
from connect.EEGInfoDataBean import EEGInfoDataBean
from connect.EEGDataTransformHelper import channelCount, dataCount, parseEEGDataNoFlag
import connect.ByteUtil as ByteUtil


def parse_packet(raw_data: bytes) -> EEGPacketData:
        raw_str = ByteUtil.bytes_to_hex_string(raw_data)
        frame_header = raw_data[:2]
        info = raw_data[11:203]  # 11 bytes for frame header, 192 bytes for info
        return EEGPacketData(raw_str, frame_header, info)

def parse_info_data(info_data: bytes, comb: bool, low_pass: bool) -> EEGInfoDataBean:
        eeg_list = info_data[:channelCount * 3 * dataCount]
        eeg_data = parseEEGDataNoFlag(eeg_list, comb, low_pass)
        return EEGInfoDataBean(eeg_data)
