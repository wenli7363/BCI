import logging
import serial
import serial.tools.list_ports
from com.example.member.eeg import EEGDataDriver

logger = logging.getLogger(__name__)

class EEGSerialPortManager:
    SERIAL_PORT_NAME = "COM7"  # 串口名
    SERIAL_BAUD_RATE = 460800  # 波特率
    EEG_DATA_PACKET_LENGTH = 204  # 数据包长度
    SEND_DATA_ORDER = "55AA0201"  # 发送数据的命令字符串

    @staticmethod
    def open_serial_port():
        m_serial_port = None
        ports = list(serial.tools.list_ports.comports())  # 获取串口列表
        for port in ports:
            if port.device == EEGSerialPortManager.SERIAL_PORT_NAME:
                m_serial_port = serial.Serial(port.device, baudrate=EEGSerialPortManager.SERIAL_BAUD_RATE)
                logger.info(f"Opened serial port: {m_serial_port.name}")
                break
        if m_serial_port is None:
            logger.error(f"Could not find serial port: {EEGSerialPortManager.SERIAL_PORT_NAME}")
        return m_serial_port

    @staticmethod
    def config_serial_port(m_serial_port):
        def serial_event(event):
            if event.event_type == serial.Serial.LISTENING_EVENT_DATA_RECEIVED:
                read_buffer = event.data
                bytes_num = len(read_buffer)
                if bytes_num == EEGSerialPortManager.EEG_DATA_PACKET_LENGTH:
                    EEGDataDriver().parse_sampling(read_buffer)
                else:
                    packet_num = bytes_num // EEGSerialPortManager.EEG_DATA_PACKET_LENGTH
                    for i in range(packet_num):
                        single_packet_data = read_buffer[i * EEGSerialPortManager.EEG_DATA_PACKET_LENGTH:
                                                         (i + 1) * EEGSerialPortManager.EEG_DATA_PACKET_LENGTH]
                        EEGDataDriver().parse_sampling(single_packet_data)

        m_serial_port.reset_input_buffer()
        m_serial_port.reset_output_buffer()

        m_serial_port.close()
        m_serial_port.open()
        m_serial_port.write(EEGSerialPortManager.SEND_DATA_ORDER.encode('utf-8'))

        m_serial_port.reset_input_buffer()
        m_serial_port.reset_output_buffer()

        m_serial_port.read(1)
        m_serial_port.reset_input_buffer()
        m_serial_port.reset_output_buffer()

        m_serial_port.close()

    @staticmethod
    def close_serial_port(m_serial_port):
        m_serial_port.close()

    @staticmethod
    def is_serial_port_open(m_serial_port):
        return m_serial_port.is_open
