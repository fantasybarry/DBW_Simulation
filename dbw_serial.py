"""Jetson -> DBW adapter serial link, carrying CAN 0x740 (steering-wheel angle + target speed)."""
import serial

DBW_PORT = "/dev/ttyUSB1"


def encode_0x740(angle_deg, speed_mph):
    """-> bytes to write to the adapter for one 0x740 frame. angle: steering-wheel deg, + = RIGHT."""
    raise NotImplementedError("dbw_serial.encode_0x740: fill in the adapter framing + 0x740 byte layout")


class DBW:
    def __init__(self, port=None, baud=115200):
        self.port = port or DBW_PORT
        self.ser = serial.Serial(self.port, baud, timeout=0.1)

    def steer(self, angle_deg, accel=0.0):
        """angle_deg: steering-wheel angle (+ = right). accel: target speed (mph); 0 = stop."""
        self.ser.write(encode_0x740(angle_deg, accel))

    def close(self):
        try:
            self.steer(0.0, 0.0)
        except (NotImplementedError, serial.SerialException):
            pass
        self.ser.close()
