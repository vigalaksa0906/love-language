import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Muat gambar love dan i love you dengan transparansi
love_img = cv2.imread('heart1.png', cv2.IMREAD_UNCHANGED)  # Membaca gambar dengan channel alpha
i_love_you_img = cv2.imread('ily.png', cv2.IMREAD_UNCHANGED)

# Fungsi untuk menemukan posisi tangan kanan dan kiri
def get_hand_positions(hand_landmarks_list, frame_width, frame_height):
    positions = {'left': None, 'right': None}
    for landmarks in hand_landmarks_list:
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        
        # Temukan rata-rata posisi
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        
        # Tentukan tangan berdasarkan posisi rata-rata
        if avg_x < 0.5:  # Posisi tangan kiri (kiri dari frame)
            positions['left'] = (avg_x * frame_width, avg_y * frame_height)
        else:  # Posisi tangan kanan (kanan dari frame)
            positions['right'] = (avg_x * frame_width, avg_y * frame_height)
    
    return positions

# Buka kamera
cap = cv2.VideoCapture(0)

# Loop utama
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Temukan posisi tangan
        hand_positions = get_hand_positions(results.multi_hand_landmarks, frame.shape[1], frame.shape[0])
        
        # Pastikan kedua tangan terdeteksi
        if hand_positions['left'] and hand_positions['right']:
            # Hitung jarak antara dua tangan
            left_x, left_y = hand_positions['left']
            right_x, right_y = hand_positions['right']
            distance = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)
            
            # Pilih gambar sesuai jarak
            if distance < 350:
                display_img = love_img
            else:
                display_img = i_love_you_img
            
            # Skala gambar
            img_height, img_width = display_img.shape[:2]
            scale_factor = min(max(0.1, distance / 350), 1)  # Skala gambar
            resized_img = cv2.resize(display_img, (int(img_width * scale_factor), int(img_height * scale_factor)))
            
            # Temukan posisi tengah antara dua tangan
            middle_x = int((left_x + right_x) / 2)
            middle_y = int((left_y + right_y) / 2)
            
            # Tentukan posisi gambar
            x_offset = middle_x - resized_img.shape[1] // 2
            y_offset = middle_y - resized_img.shape[0] // 2
            
            # Pastikan gambar yang di-resize cocok dengan area di frame
            x_offset = max(0, min(x_offset, frame.shape[1] - resized_img.shape[1]))
            y_offset = max(0, min(y_offset, frame.shape[0] - resized_img.shape[0]))
            
            # Pisahkan channel alpha
            if resized_img.shape[2] == 4:
                bgr_img = resized_img[:, :, :3]
                alpha_channel = resized_img[:, :, 3] / 255.0
                for c in range(3):
                    frame[y_offset:y_offset+resized_img.shape[0], x_offset:x_offset+resized_img.shape[1], c] = \
                        (alpha_channel * bgr_img[:, :, c] + (1 - alpha_channel) * frame[y_offset:y_offset+resized_img.shape[0], x_offset:x_offset+resized_img.shape[1], c]).astype(np.uint8)
            else:
                frame[y_offset:y_offset+resized_img.shape[0], x_offset:x_offset+resized_img.shape[1]] = resized_img

    # Tampilkan hasil
    cv2.imshow('Love bgtt inii mah!!', frame)
    
    # Periksa apakah jendela ditutup atau tombol 'q' ditekan
    if cv2.getWindowProperty('Love bgtt inii mah!!', cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
