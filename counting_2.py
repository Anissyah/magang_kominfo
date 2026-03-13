import os
import cv2
import time
import numpy as np
import mysql.connector
from datetime import datetime
from threading import Thread, Lock
from ultralytics import YOLO
from pynput import keyboard

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded")

# ================= DATABASE =================

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="traffic_monitoring"
)

cursor = db.cursor()

def save_to_db(object_name, direction, device):

    query = """
    INSERT INTO object_real_times
    (object,count,start_date,end_date,created_at,email,direction,device_id)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """

    now = datetime.now()

    values = (
        object_name,
        1,
        now,
        now,
        now,
        "ai@traffic.com",
        direction,
        device
    )

    cursor.execute(query, values)
    db.commit()

# ================= AREA MASUK =================

area_masuk = {

"CCTV1":[(517,50),(572,77),(198,169),(179,129)],
"CCTV2":[(623,73),(628,182),(356,169),(428,48)],
"CCTV3":[(381,77),(438,91),(324,146),(270,116)],
"CCTV4":[(587,130),(407,102),(525,42),(632,69)]

}

# ================= AREA KELUAR =================

area_keluar = {

"CCTV1":[(414,129),(474,182),(637,103),(616,68)],
"CCTV2":[(619,193),(632,354),(48,352),(62,165)],
"CCTV3":[(295,170),(492,259),(354,350),(136,230)],
"CCTV4":[(61,127),(168,168),(472,37),(414,31)]

}

# ================= CCTV STREAM =================

cctv_feeds = {

"CCTV1":"https://cctv.jogjaprov.go.id/cctv-proxy/atcs/DemenGlagah_TC.stream/playlist.m3u8",
"CCTV2":"https://cctv.jogjaprov.go.id/cctv-proxy/atcs/Prambanan_TC.stream/playlist.m3u8",
"CCTV3":"https://cctv.jogjaprov.go.id/cctv-proxy/cctv-kominfogk/TuguSelamatDatangPatuk.stream/playlist.m3u8",
"CCTV4":"https://mam.jogjaprov.go.id:1937/cctv-kominfosleman/Perempatan_Beran1.stream/playlist.m3u8"

}

vehicle_classes = ["car","motorcycle","bus","truck"]

stop_program = False

# ================= DETEKSI TOMBOL Q =================

def on_press(key):

    global stop_program

    try:
        if key.char == 'q':
            print("\nProgram dihentikan...")
            stop_program = True
            return False
    except:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# ================= STREAM CLASS =================

class VideoStream:

    def __init__(self, src):

        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        self.frame = None
        self.ret = False

        self.lock = Lock()
        self.stopped = False

        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):

        while not self.stopped:

            try:

                if not self.cap.isOpened():

                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                    time.sleep(2)
                    continue

                ret, frame = self.cap.read()

                if not ret:

                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                    continue

                with self.lock:

                    self.ret = ret
                    self.frame = frame

            except:

                print("Stream error:", self.src)
                time.sleep(2)

    def read(self):

        with self.lock:

            if self.frame is None:
                return False,None

            return self.ret,self.frame.copy()

    def stop(self):

        self.stopped = True
        self.cap.release()

streams = {name: VideoStream(url) for name,url in cctv_feeds.items()}

# ================= DATA COUNT =================

vehicle_masuk = {

name:{cls:0 for cls in vehicle_classes}
for name in cctv_feeds

}

vehicle_keluar = {

name:{cls:0 for cls in vehicle_classes}
for name in cctv_feeds

}

track_history = {name:{} for name in cctv_feeds}

counted_masuk = {name:set() for name in cctv_feeds}
counted_keluar = {name:set() for name in cctv_feeds}

last_print = time.time()

# ================= CEK POINT DALAM POLYGON =================

def point_in_polygon(point, polygon):

    polygon_np = np.array(polygon, np.int32)
    result = cv2.pointPolygonTest(polygon_np, point, False)

    return result >= 0

# ================= MAIN PROCESS =================

while not stop_program:

    for name in cctv_feeds:

        ret,frame = streams[name].read()

        if not ret or frame is None:
            continue

        frame = cv2.resize(frame,(640,360))

        results = model.predict(frame,conf=0.4,verbose=False)

        for box in results[0].boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label not in vehicle_classes:
                continue

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            point = (cx,cy)

            track_id = f"{cx//30}_{cy//30}"

            # ================= AREA MASUK =================

            if point_in_polygon(point, area_masuk[name]):

                if track_id not in counted_masuk[name]:

                    vehicle_masuk[name][label] += 1
                    counted_masuk[name].add(track_id)

                    save_to_db(label,1,name)

            # ================= AREA KELUAR =================

            if point_in_polygon(point, area_keluar[name]):

                if track_id not in counted_keluar[name]:

                    vehicle_keluar[name][label] += 1
                    counted_keluar[name].add(track_id)

                    save_to_db(label,2,name)

    # ================= PRINT REALTIME =================

    if time.time() - last_print > 1:

        os.system("cls" if os.name=="nt" else "clear")

        print("===== KENDARAAN MASUK JOGJA =====")

        for name in vehicle_masuk:

            print("\n",name)

            total = 0

            for v,c in vehicle_masuk[name].items():

                print(v,":",c)
                total += c

            print("TOTAL:",total)

        print("\n===== KENDARAAN KELUAR JOGJA =====")

        for name in vehicle_keluar:

            print("\n",name)

            total = 0

            for v,c in vehicle_keluar[name].items():

                print(v,":",c)
                total += c

            print("TOTAL:",total)

        last_print = time.time()

# ================= STOP STREAM =================

for stream in streams.values():
    stream.stop()