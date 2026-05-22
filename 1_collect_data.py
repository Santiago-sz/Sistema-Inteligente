import cv2
import os

IMG_SIZE = 64

os.makedirs('data/movement', exist_ok=True)
os.makedirs('data/no_movement', exist_ok=True)

cap = cv2.VideoCapture(0)
prev_frame = None
count = {'movement': 0, 'no_movement': 0}

print("M = capturar MOVIMIENTO | N = SIN movimiento | Q = salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)

        display = frame.copy()
        cv2.putText(display, f"MOV: {count['movement']}  |  SIN MOV: {count['no_movement']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Recoleccion de datos', display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('m'), ord('M')):
            cv2.imwrite(f'data/movement/{count["movement"]}.jpg', diff)
            count['movement'] += 1
            print(f"Movimiento #{count['movement']} guardado")

        elif key in (ord('n'), ord('N')):
            cv2.imwrite(f'data/no_movement/{count["no_movement"]}.jpg', diff)
            count['no_movement'] += 1
            print(f"Sin movimiento #{count['no_movement']} guardado")

        elif key == ord('q'):
            break

    prev_frame = gray

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal recolectado -> Movimiento: {count['movement']} | Sin movimiento: {count['no_movement']}")
