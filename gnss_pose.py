"""On-car pose sources. Each runs a serial reader thread; pose() -> (x, y, yaw) or None (no fresh data).

GNSSPose : Septentrio NMEA over USB (GGA position + fix quality, RMC speed + course).
FusedPose: GNSSPose position + VectorNav gyro-integrated heading (valid at standstill, unlike GPS
           course), pulled slowly toward the GPS course while moving to cancel gyro drift.

Frames: x east, y north (m from the path origin, path_io.ll_to_xy); yaw ENU rad (0 = east, + = CCW).
Pose is reported at the REAR AXLE (pure pursuit's reference point), corrected for antenna offset.
"""
import math
import threading
import time

import serial

from path_io import ll_to_xy

GPS_PORT       = "/dev/ttyACM0"
IMU_PORT       = "/dev/ttyUSB0"
ANT_FWD_M      = 0.0      # GPS antenna ahead of the rear axle (m) -- measure on the car
ANT_LEFT_M     = 0.0      # GPS antenna left of the car centerline (m)
GPS_TIMEOUT_S  = 1.0      # no GGA fix for this long -> pose None (pure_pursuit commands a stop)
COURSE_MIN_MPS = 1.0      # GPS course is noise below this speed

IMU_YAW_SIGN   = -1       # VectorNav body z points DOWN (+gz = turning right); ENU yaw + = left.
                          # Flip if the IMU is mounted upside down.
IMU_TIMEOUT_S  = 0.3
STILL_MPS      = 0.05     # GPS speed under this = parked: hold heading, learn the gyro bias
K_COURSE       = 0.2      # 1/s: how fast GPS course corrects IMU heading while moving (0 = gyro only)

FIX_NAMES = {0: "NONE", 1: "GPS", 2: "DGPS", 4: "RTK", 5: "RTK-FLOAT", 6: "DR"}


def nmea_fields(line):
    """'$...*CS' -> list of fields if the XOR checksum matches, else None (serial noise / partial line)."""
    if not line.startswith("$") or "*" not in line:
        return None
    body, _, cs = line[1:].partition("*")
    calc = 0
    for c in body:
        calc ^= ord(c)
    try:
        if int(cs[:2], 16) != calc:
            return None
    except ValueError:
        return None
    return ("$" + body).split(",")


def _deg(v, hemi):
    """NMEA ddmm.mmmm / dddmm.mmmm -> signed decimal degrees."""
    d = int(float(v) / 100)
    return (d + (float(v) - 100 * d) / 60.0) * (-1 if hemi in ("S", "W") else 1)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _rear_axle(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x - ANT_FWD_M * c + ANT_LEFT_M * s, y - ANT_FWD_M * s - ANT_LEFT_M * c, yaw


def _start_reader(ser, handle):
    """Thread: read lines, hand checksum-valid NMEA-style field lists to handle()."""
    def run():
        while ser.is_open:
            try:
                f = nmea_fields(ser.readline().decode("ascii", "ignore").strip())
            except (serial.SerialException, OSError, TypeError, AttributeError):
                break                                             # port closed / unplugged
            if f:
                try:
                    handle(f)
                except (ValueError, IndexError):
                    pass                                          # empty/malformed field (e.g. no fix yet)
    threading.Thread(target=run, daemon=True).start()


class GNSSPose:
    def __init__(self, port=None, baud=115200):
        self.port = port or GPS_PORT
        self.ser = serial.Serial(self.port, baud, timeout=0.5)
        self.fix, self.ll, self.t_fix = 0, None, 0.0
        self.speed, self.course = 0.0, None
        self.lat0 = self.lon0 = None
        _start_reader(self.ser, self._gps)

    def _gps(self, f):
        kind = f[0][3:]                                           # $GPGGA / $GNGGA -> GGA
        if kind == "GGA":
            self.fix = int(f[6] or 0)
            if self.fix > 0:
                self.ll = (_deg(f[2], f[3]), _deg(f[4], f[5]))
                self.t_fix = time.monotonic()
        elif kind == "RMC" and f[2] == "A":
            self.speed = float(f[7] or 0) * 0.514444              # knots -> m/s
            if self.speed > COURSE_MIN_MPS and f[8]:
                self.course = _wrap(math.radians(90.0 - float(f[8])))   # compass -> ENU

    def ready(self):
        return self.fix > 0 and self.ll is not None

    def fix_name(self):
        return FIX_NAMES.get(self.fix, str(self.fix))

    def set_origin(self, lat0=None, lon0=None):
        self.lat0, self.lon0 = (lat0, lon0) if lat0 is not None else self.ll

    def xy(self):
        """Antenna (x, y) from the latest fix, or None if there's no fresh fix."""
        if self.ll is None or self.lat0 is None or self.fix == 0 or time.monotonic() - self.t_fix > GPS_TIMEOUT_S:
            return None
        return ll_to_xy(*self.ll, self.lat0, self.lon0)

    def pose(self):
        p = self.xy()
        return None if p is None or self.course is None else _rear_axle(*p, self.course)

    def close(self):
        self.ser.close()


class FusedPose(GNSSPose):
    """GNSS position + VectorNav heading. Expects the VectorNav default ASCII output ($VNYMR)."""
    # ponytail: VectorNav ASCII only; Xsens (MTData2) / MicroStrain (MIP) are binary and need their own _imu()
    def __init__(self, gps_port=None, imu_port=None, baud=115200):
        super().__init__(gps_port)
        self.imu_port = imu_port or IMU_PORT
        self.imu = serial.Serial(self.imu_port, baud, timeout=0.5)
        self.yaw, self.bias, self.t_imu = None, 0.0, None
        self.gyro_yaw = 0.0          # pure gyro heading, arbitrary zero, never GPS-corrected (lidar_pose uses its deltas)
        _start_reader(self.imu, self._imu)

    def _imu(self, f):
        if f[0] != "$VNYMR":         # yaw,pitch,roll, mag xyz, accel xyz, gyro xyz (rad/s)
            return
        now = time.monotonic()
        rate = IMU_YAW_SIGN * float(f[12])                        # ENU yaw rate
        dt, self.t_imu = (now - self.t_imu if self.t_imu else 0.0), now
        if self.speed < STILL_MPS:                                # parked: heading can't change
            self.bias += 0.01 * (rate - self.bias)
            return
        self.gyro_yaw += (rate - self.bias) * dt
        if self.yaw is None:                                      # never seeded: adopt GPS course once moving
            self.yaw = self.course
            return
        self.yaw = _wrap(self.yaw + (rate - self.bias) * dt)
        if self.speed > COURSE_MIN_MPS and self.course is not None:
            self.yaw = _wrap(self.yaw + K_COURSE * dt * _wrap(self.course - self.yaw))

    def seed_from_path(self, path, heading_deg=None):
        """Set the standstill starting heading. -> description, or None if it can't be determined."""
        if heading_deg is not None:
            self.yaw = math.radians(90.0 - heading_deg)
            return f"heading {heading_deg:.0f} deg (from --heading)"
        p = self.xy()
        if p is None or math.hypot(p[0] - path[0][0], p[1] - path[0][1]) > 3.0:
            return None
        self.yaw = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        return f"heading {(90 - math.degrees(self.yaw)) % 360:.0f} deg (path start direction)"

    def pose(self):
        p = self.xy()
        if p is None or self.yaw is None or self.t_imu is None or time.monotonic() - self.t_imu > IMU_TIMEOUT_S:
            return None
        return _rear_axle(*p, self.yaw)

    def close(self):
        super().close()
        self.imu.close()
