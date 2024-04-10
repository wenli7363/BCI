import serial
import threading
import logging
from queue import Queue
from EEGDataDriver import EEGDataDriver

eeg_driver = EEGDataDriver()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERIAL_PORT_NAME = "COM3"  # 串口名
SERIAL_BAUD_RATE = 460800  # 波特率
EEG_DATA_PACKET_LENGTH = 204  # 数据包长度
SEND_DATA_ORDER = b"\x55\xAA\x02\x01"  # 发送数据的命令字节串

message_queue = Queue()

class EEGSerialPortManager:
    def __init__(self):
        self.serial_port = None
        self.serial_thread = None

    def open_serial_port(self):
        try:
            self.serial_port = serial.Serial(SERIAL_PORT_NAME, SERIAL_BAUD_RATE)
            if self.serial_port.is_open:
                logger.info("Serial port opened successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to open serial port: {e}")
        return False

    def config_serial_port(self):
        if self.serial_port is not None and self.serial_port.is_open:
            self.serial_thread = threading.Thread(target=self.serial_listener,args=(message_queue,))
            self.serial_thread.daemon = True
            self.serial_thread.start()
            logger.info("Serial port listener thread started")
        else:
            logger.error("Serial port is not open, cannot configure")

    def serial_listener(self,q):
        while self.is_serial_port_open():
            try:
                if self.serial_port.in_waiting >= EEG_DATA_PACKET_LENGTH:
                    read_buffer = self.serial_port.read(EEG_DATA_PACKET_LENGTH)
                    # self.parse_sampling(read_buffer)
                    eeg_driver.parse_sampling(read_buffer)
                    # q.put(eeg_driver.parse_sampling(read_buffer))

                    # q.put(read_buffer)
            except Exception as e:
                logger.error(f"Serial port listener error: {e}")
                if not self.is_serial_port_open():
                    self.open_serial_port()

    @staticmethod
    def parse_sampling(data):
        # Placeholder for parsing EEG data
        logger.info(f"Parsing EEG data: {data}")

    def close_serial_port(self):
        if self.serial_port     is not None and self.serial_port.is_open:
            self.serial_port.close()
            logger.info("Serial port closed")
        else:
            logger.error("Serial port is not open, cannot close")

    def send_data(self, data):
        if self.serial_port is not None and self.serial_port.is_open:
            try:
                self.serial_port.write(data)
                logger.info("Data sent successfully")
                return len(data)
            except Exception as e:
                logger.error(f"Failed to send data: {e}")
        else:
            logger.error("Serial port is not open, cannot send data")
        return 0

    def request_data(self):
        if self.send_data(SEND_DATA_ORDER) == 0:
            logger.error("Failed to send data: 55AA0201")
        else:
            logger.info("Data sent successfully: 55AA0201")

    def is_serial_port_open(self):
        if self.serial_port is not None:
            return self.serial_port.is_open
        return False


# if __name__ == "__main__":
#     eeg_serial_manager = EEGSerialPortManager()
#     if eeg_serial_manager.open_serial_port():
#         eeg_serial_manager.config_serial_port()
#         # Example usage:
#         eeg_serial_manager.request_data()
#         # if eeg_serial_manager.is_serial_port_open():
#         #     # Do something

#         # else:
#         #     # Handle serial port not open
#     else:
#         logger.error("Failed to open serial port")
