import time
import winsound

def beep_every_5_seconds():
    while True:
        # 播放蜂鸣声
        winsound.Beep(1000, 1000)  # 第一个参数是频率，第二个参数是持续时间（毫秒）
        # 等待5秒
        time.sleep(5)

if __name__ == "__main__":
    beep_every_5_seconds()
