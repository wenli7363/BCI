import byte_util  # 假设您已经实现了一个类似于Java ByteUtil的Python工具类
import eeg_data_transform_helper  # 假设您已经实现了一个类似于Java EEGDataTransformHelper的Python工具类

class EEGPacketDataParser:
    @staticmethod
    def parser_packet(raw_data):
        # 假设raw_data是一个bytes类型的数据，包含完整的EEG数据包
        raw_str = byte_util.bytes_to_hex_string(raw_data)  # 将原始字节数据转换为十六进制字符串
        frame_header = raw_data[:2]  # 取出数据帧头
        info = raw_data[11:]  # 取出数据内容

        return EEGPacketDataBean(raw_str, frame_header, info)

    @staticmethod
    def parser_info_data(info_data, comb=False, low_pass=False):
        # 假设info_data是一个bytes类型的数据，包含EEG信息数据
        eeg_list = info_data[:eeg_data_transform_helper.channel_count * 3 * eeg_data_transform_helper.data_count]
        eeg_data = eeg_data_transform_helper.parse_eeg_data_no_flag(eeg_list, comb, low_pass)
        return EEGInfoDataBean(eeg_data)