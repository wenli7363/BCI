import EEGPacketDataBean
import EEGInfoDataBean
import EEGDataTransformHelper
import ByteUtil

class EEGPacketDataParser:

    @staticmethod
    def parse_packet(raw_data: bytes) -> EEGPacketDataBean:
        raw_str = ByteUtil.bytes_to_hex_string(raw_data)
        frame_header = raw_data[:2]
        info = raw_data[11:203]  # 11 bytes for frame header, 192 bytes for info
        return EEGPacketDataBean(raw_str, frame_header, info)

    @staticmethod
    def parse_info_data(info_data: bytes, comb: bool, low_pass: bool) -> EEGInfoDataBean:
        eeg_list = info_data[:EEGDataTransformHelper.channelCount * 3 * EEGDataTransformHelper.dataCount]
        eeg_data = EEGDataTransformHelper.parse_eeg_data_no_flag(eeg_list, comb, low_pass)
        return EEGInfoDataBean(eeg_data)
