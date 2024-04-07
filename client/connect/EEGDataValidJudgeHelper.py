class EEGDataValidJudgeHelper:
    FRAME_HEADER = bytes([0xAA, 0x55])  # 帧头

    @staticmethod
    def is_valid_of_frame_header(frame_header):
        return frame_header == EEGDataValidJudgeHelper.FRAME_HEADER