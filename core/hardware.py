import serial
import time
import numpy as np
import cv2 as cv
from PIL import Image

try:
    from picamera2.previews.qt import QPicamera2
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except Exception:
    QPicamera2 = None
    Picamera2 = None
    PICAM_AVAILABLE = False

class ESP32Controller:
    def __init__(self):
        self.serial_conn = None
        self.available = False
        self._connect()

    def _connect(self):
        ports = ['/dev/ttyUSB0', '/dev/ttyACM0']
        for port in ports:
            try:
                self.serial_conn = serial.Serial(port, 115200, timeout=1)
                self.available = True
                print(f"ESP32 Terhubung via {port}!")
                time.sleep(2)
                break
            except Exception:
                continue
        if not self.available:
            print("ESP32 Tidak Terdeteksi. Program jalan dalam Mode Simulasi Motor.")

    def send_command(self, direction_char, steps):
        if self.available and self.serial_conn and self.serial_conn.is_open:
            pesan = f"{direction_char}{steps}\n"
            try:
                self.serial_conn.write(pesan.encode('utf-8'))
                print(f"📡 Mengirim ke ESP32: {pesan.strip()}")
            except Exception as e:
                print(f"Gagal mengirim perintah serial: {e}")
        else:
            print(f"Mode Simulasi: Motor {direction_char} {steps} langkah")

    def stop(self):
        if self.available and self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write("S0\n".encode('utf-8'))
            print("Motor Stop")

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()


class MagnificationSensor:
    # Spesifikasi sensor kamera RPi Camera Module V3
    SENSOR_WIDTH_MM  = 6.287
    SENSOR_HEIGHT_MM = 4.712

    def __init__(self, reference_distance=160.0, reference_magnification=1000.0):
        """
        Args:
            reference_distance      : jarak referensi kalibrasi (mm)
            reference_magnification : magnifikasi pada jarak referensi (x)
        """
        self.reference_distance      = reference_distance
        self.reference_magnification = reference_magnification

        self._sensor = None
        self._mode   = "simulation"
        self._init_sensor()

    # ------------------------------------------------------------------ #
    #  Inisialisasi hardware                                               #
    # ------------------------------------------------------------------ #
    def _init_sensor(self):
        try:
            import board
            import busio
            import adafruit_vl53l0x
            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self._sensor.measurement_timing_budget = 200000
            self._mode = "vl53l0x"
            print("Sensor VL53L0X terdeteksi via I2C")
        except Exception as e:
            print(f"VL53L0X tidak terdeteksi ({e}), mode simulasi aktif")
            self._mode = "simulation"

    # ------------------------------------------------------------------ #
    #  Baca jarak                                                          #
    # ------------------------------------------------------------------ #
    def read_distance(self):
        """Kembalikan jarak dalam cm. -1.0 jika gagal."""
        if self._mode == "vl53l0x":
            try:
                mm = self._sensor.range
                if mm <= 0 or mm > 8000:
                    return -1.0
                return mm / 10.0          # konversi mm → cm
            except Exception as e:
                print(f"Gagal baca sensor: {e}")
                return -1.0
        return 15.4                       # nilai simulasi default (cm)

    # ------------------------------------------------------------------ #
    #  Hitung magnifikasi dari jarak                                       #
    # ------------------------------------------------------------------ #
    def calculate_magnification(self, distance_cm: float) -> float:
        """
        Hitung magnifikasi berdasarkan jarak (cm).

        Rumus: M = (d_ref × M_ref) / d
        Hubungan invers: semakin dekat → magnifikasi semakin besar.

        Returns:
            float: nilai magnifikasi (x), atau -1.0 jika input tidak valid
        """
        if distance_cm <= 0:
            return -1.0
        distance_mm = distance_cm * 10.0
        return (self.reference_distance * self.reference_magnification) / distance_mm

    # ------------------------------------------------------------------ #
    #  Hitung Field of View dari magnifikasi                               #
    # ------------------------------------------------------------------ #
    def calculate_fov(self, magnification: float):
        """
        Hitung Field of View (FOV) berdasarkan magnifikasi.

        FOV = Ukuran sensor fisik / Magnifikasi total

        Returns:
            tuple (fov_width_mm, fov_height_mm), atau (-1.0, -1.0) jika invalid
        """
        if magnification <= 0:
            return -1.0, -1.0
        fov_w = self.SENSOR_WIDTH_MM  / magnification
        fov_h = self.SENSOR_HEIGHT_MM / magnification
        return fov_w, fov_h

    # ------------------------------------------------------------------ #
    #  Baca semua sekaligus (convenience method)                           #
    # ------------------------------------------------------------------ #
    def read_all(self):
        """
        Baca jarak lalu hitung magnifikasi dan FOV sekaligus.

        Returns:
            dict dengan key: distance_cm, distance_mm,
                             magnification, fov_width_mm, fov_height_mm
        """
        distance_cm   = self.read_distance()
        magnification = self.calculate_magnification(distance_cm)
        fov_w, fov_h  = self.calculate_fov(magnification)

        return {
            "distance_cm"   : distance_cm,
            "distance_mm"   : distance_cm * 10.0 if distance_cm > 0 else -1.0,
            "magnification" : magnification,
            "fov_width_mm"  : fov_w,
            "fov_height_mm" : fov_h,
        }

    # ------------------------------------------------------------------ #
    #  Kalibrasi ulang                                                     #
    # ------------------------------------------------------------------ #
    def calibrate(self, current_distance_cm: float, known_magnification: float):
        """
        Kalibrasi ulang sensor dengan nilai yang diketahui.

        Args:
            current_distance_cm : jarak aktual saat ini (cm)
            known_magnification : magnifikasi yang diketahui pada jarak tersebut
        """
        if current_distance_cm > 0 and known_magnification > 0:
            self.reference_distance      = current_distance_cm * 10.0  # simpan dalam mm
            self.reference_magnification = known_magnification
            print(f"✅ Kalibrasi berhasil: {current_distance_cm:.1f}cm = {known_magnification:.0f}x")


class CameraSystem:
    def __init__(self):
        self.using_picam = PICAM_AVAILABLE
        self.picam2 = None
        self.qpicamera2 = None
        self.cap = None

        if self.using_picam:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(self.picam2.create_preview_configuration({"size": (480, 270)}))
                self.qpicamera2 = QPicamera2(self.picam2, width=480, height=270, keep_ar=True)
            except Exception:
                self.using_picam = False

        if not self.using_picam:
            self.cap = cv.VideoCapture(0)

    def start_camera(self):
        if self.using_picam and self.picam2:
            try: self.picam2.start()
            except Exception: pass

    def capture_image(self, save_path, signal_callback=None):
        if self.using_picam and self.picam2:
            cfg = self.picam2.create_still_configuration(main={"size": (480, 270)})
            self.picam2.switch_mode_and_capture_file(cfg, save_path, signal_function=signal_callback)
            return True
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv.imwrite(save_path, frame)
                return True
        return False

    def get_opencv_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret: return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return None

    def close(self):
        if self.using_picam and self.picam2:
            try: self.picam2.stop()
            except Exception: pass
        if self.cap and self.cap.isOpened():
            self.cap.release()
