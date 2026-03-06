#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

# Kendi YOLOv8 drone model blob dosyamızın yolu
nnPath = str((Path(__file__).parent / Path('models/drone_yolov8n_640x352.blob')).resolve().absolute())
if not Path(nnPath).exists():
    raise FileNotFoundError(f'Required blob not found at path: {nnPath}')

# Sadece 1 sınıfımız var: drone
labelMap = ["drone"]

syncNN = True

# Pipeline oluştur
pipeline = dai.Pipeline()

# Kaynaklar ve çıkışlar
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Kamera ayarları
camRgb.setPreviewSize(352, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# YOLO ağ ayarları
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(1)              # TEK SINIF: drone
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Bağlantılar
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Cihaza bağlan ve pipeline'ı başlat
with dai.Device(pipeline) as device:

    # Kuyruklar
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)
    frame = None

    # NN çıktısı 0..1 aralığında, frame boyutuyla çarpıp piksele çeviriyoruz
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            # sınıf adı ve güven
            label = labelMap[detection.label] if detection.label < len(labelMap) else str(detection.label)
            cv2.putText(frame, label, (bbox[0] + 10, bbox[1] + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%",
                        (bbox[0] + 10, bbox[1] + 40),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        cv2.imshow(name, frame)

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime + 1e-6)),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                color2,
            )

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
