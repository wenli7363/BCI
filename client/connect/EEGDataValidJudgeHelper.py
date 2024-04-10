# 有效性校验，检查帧头是否正确

FRAME_HEADER = bytes([0xAA, 0x55])  # 帧头

def is_valid_of_frame_header(frame_header):
    return frame_header == FRAME_HEADER
