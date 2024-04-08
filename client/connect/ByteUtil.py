import binascii

# 将字节数据转换为十六进制字符串
def bytes_to_hex_string(data):
    return binascii.hexlify(data).decode('utf-8')


# class ByteUtil:
#     @staticmethod
#     def bytes_to_hex_string(data: bytes) -> str:
#         return ''.join([format(byte, '02X') for byte in data])