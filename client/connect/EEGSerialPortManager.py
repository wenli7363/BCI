import serial
import threading
import logging
import EEGDataDriver

# 设置日志记录器
logger = logging.getLogger(__name__)

class EEGSerialPortManager:
    SERIAL_PORT_NAME = "COM3"  # 串口名
    SERIAL_BAUD_RATE = 460800  # 波特率
    EEG_DATA_PACKET_LENGTH = 204  # 数据包长度
    SEND_DATA_ORDER = "55AA0201"  # 发送数据的命令字符串

    def __init__(self):
        self.serial_port = None

    # 尝试打开指定的串口。
    def open_serial_port(self):
        try:
            # self.serial_port = serial.Serial(port=self.SERIAL_PORT_NAME, baudrate=self.SERIAL_BAUD_RATE)
            self.serial_port = serial.Serial(port=self.SERIAL_PORT_NAME,
                                 baudrate=self.SERIAL_BAUD_RATE,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE)
            logger.info(f"Opened serial port: {self.SERIAL_PORT_NAME}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port: {e}")
            return False

    # 配置串口
    def config_serial_port(self):
        if self.serial_port is not None and self.serial_port.isOpen():
            thread = threading.Thread(target=self.listen_to_serial_port)
            thread.daemon = True
            thread.start()
        else:
            logger.error("Serial port is not open.")

    # 监听串口数据
    def listen_to_serial_port(self):
        while True:
            try:
                if self.serial_port.in_waiting > 0:
                    read_buffer = self.serial_port.read(self.serial_port.in_waiting)
                    bytes_num = len(read_buffer)
                    # 如果是完整的数据包，就处理
                    if bytes_num == self.EEG_DATA_PACKET_LENGTH:
                        EEGDataDriver.parse_sampling(read_buffer)  # 开始处理脑电数据
                    else:
                        packet_num = bytes_num // self.EEG_DATA_PACKET_LENGTH
                        for i in range(packet_num):
                            single_packet_data = read_buffer[i * self.EEG_DATA_PACKET_LENGTH:
                                                             (i + 1) * self.EEG_DATA_PACKET_LENGTH]
                            EEGDataDriver.parse_sampling(single_packet_data)  # 开始处理脑电数据
            except serial.SerialException as e:
                logger.error(f"Serial port error: {e}") 

    # 向串口发送数据
    def send_data(self, data):
        if self.serial_port is not None and self.serial_port.isOpen():
            try:
                data_bytes = bytes.fromhex(data)
                self.serial_port.write(data_bytes)
                logger.info(f"Sent data: {data}")
                return len(data_bytes)
            except serial.SerialException as e:
                logger.error(f"Failed to send data: {e}")
                return 0
        else:
            logger.error("Serial port is not open.")
            return 0

    # 发送一个特定的命令字符串到串口，并记录发送结果
    def request_data(self):
        if self.send_data(self.SEND_DATA_ORDER) == 0:
            logger.error("Failed to send command 55AA0201")
        else:
            logger.info("Sent command 55AA0201")

    # 关闭串口
    def close_serial_port(self):
        if self.serial_port is not None and self.serial_port.isOpen():
            self.serial_port.close()
            logger.info("Closed serial port.")
        else:
            logger.error("Serial port is not open.")

    # 判断串口是否打开
    def is_serial_port_open(self):
        return self.serial_port is not None and self.serial_port.isOpen()

# 创建EEGSerialPortManager实例
eeg_serial_port_manager = EEGSerialPortManager()
# 尝试打开串口
if eeg_serial_port_manager.open_serial_port():
    # 配置串口
    eeg_serial_port_manager.config_serial_port()
    # 发送请求数据命令
    eeg_serial_port_manager.request_data()
else:
    logger.error("Failed to open serial port.")
