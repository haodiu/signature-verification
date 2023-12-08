from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys

import torchvision.transforms as transforms
from PIL import Image
from model import SiameseNetwork
import torch.nn.functional as F
import torch

image_path_1 = ""
image_path_2 = ""

background_color = "#bab9b8"
button_verify_color = "#4979eb"

# SigComp 2011 + Chu ky thu thap
# threshold = 4.507454887376408
# model_path = "model/model_epoch_17.pt"

# SigComp 2011
model_path = "model/model_epoch_19.pt"
threshold = 5.507642486269665

image_transforms = transforms.Compose([
    transforms.Resize((155, 220)),
    transforms.ToTensor(),
])


def print_pixel_values(img):
    pixels = img.load()  # Lấy dữ liệu các pixel
    width, height = img.size

    for y in range(height):
        for x in range(width):
            pixel_value = pixels[x, y]  # Lấy giá trị của pixel tại vị trí (x, y)
            if pixel_value <= 250:
                print(f"Pixel at ({x}, {y}): {pixel_value}")


def convert_to_image_tensor(path):
    img = Image.open(path)
    img = img.convert("L")

    img = image_transforms(img)
    img = img.unsqueeze(0)  # Thêm chiều batch với kích thước 1

    return img


def load_model(model_path):
    model = SiameseNetwork()
    model.eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def upload_image_1():
    global image_path_1
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(window, "Chọn ảnh", "",
                                               "Images (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*)",
                                               options=options)

    if file_name:
        # Load ảnh đã chọn và hiển thị lên box_image_1
        image_path_1 = file_name  # Lưu đường dẫn vào biến toàn cục
        pixmap = QPixmap(image_path_1)
        box_image_1.setPixmap(pixmap)
        box_image_1.setScaledContents(True)
        box_image_1.setStyleSheet("background-color: white;")


def upload_image_2():
    global image_path_2
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(window, "Chọn ảnh", "",
                                               "Images (*.png *.jpg *.jpeg *.gif *.bmp);;All Files (*)",
                                               options=options)

    if file_name:
        # Load ảnh đã chọn và hiển thị lên box_image_2
        image_path_2 = file_name  # Lưu đường dẫn vào biến toàn cục
        pixmap = QPixmap(image_path_2)
        box_image_2.setPixmap(pixmap)
        box_image_2.setScaledContents(True)
        # Xoá border
        box_image_2.setStyleSheet("background-color: white;")


# Hàm xử lý sự kiện khi click vào nút "Xác thực"
def on_button_verify_click():
    if image_path_1 and image_path_2:
        model = load_model(model_path)

        image1 = convert_to_image_tensor(image_path_1)
        image2 = convert_to_image_tensor(image_path_2)

        print(image_path_1, image_path_2)

        output1, output2 = model(image1, image2)
        euclidean_distance = F.pairwise_distance(output1, output2).item()

        print(f"Euclidean distance: ${euclidean_distance}")
        if euclidean_distance <= threshold:
            reliability = (1 - (euclidean_distance / threshold)) * 100
            text_result.setText("MATCH")
            text_result.setStyleSheet("color: green; font-size: 50px")
            text_reliability.setText(f"Reliability: {round(reliability, 2)}%")
            text_reliability.setStyleSheet("color: black; font-size: 20")
        else:
            text_result.setText("NOT MATCH")
            text_result.setStyleSheet("color: red; font-size: 50px")
            text_reliability.setText('')

    else:
        text_result.setText("Please upload 2 images before verifying the signature")
        text_result.setStyleSheet('font-size: 16px')


app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("Signature Verification")


image = QLabel('', window)
image.setGeometry(0, 0, 340, 600)
image.setStyleSheet("background-color: red;")

image_path = "image/img.png"
image_pixmap = QPixmap(image_path)
image.setPixmap(image_pixmap)
image.setScaledContents(True)


# Nút upload ảnh 1
button_upload_image_1 = QPushButton("Upload", window)
button_upload_image_1.move(480, 260)
button_upload_image_1.setCursor(Qt.PointingHandCursor)
button_upload_image_1.clicked.connect(upload_image_1)

# Nút upload ảnh 2
button_upload_image_2 = QPushButton("Upload", window)
button_upload_image_2.move(780, 260)
button_upload_image_2.setCursor(Qt.PointingHandCursor)
button_upload_image_2.clicked.connect(upload_image_2)

# Nút xác thực
button_verify = QPushButton("Verify signature", window)
button_verify.setGeometry(580, 315, 200, 40)
button_verify.setStyleSheet(f"background-color: #6693ed; color: white; border-radius: 5px")
button_verify.setCursor(Qt.PointingHandCursor)
button_verify.clicked.connect(on_button_verify_click)

# Hộp hiển thị ảnh khi upload
box_image_1 = QLabel('Please select signature image', window)
box_image_1.setGeometry(390, 30, 280, 215)
box_image_1.setStyleSheet("background-color: white; border: 2px dotted black")
# Đặt căn giữa cho văn bản
box_image_1.setAlignment(Qt.AlignCenter)

box_image_2 = QLabel('Please select signature image', window)
box_image_2.setGeometry(690, 30, 280, 215)
box_image_2.setStyleSheet("background-color: red;")
box_image_2.setStyleSheet("background-color: white; border: 2px dotted black")
# Đặt căn giữa cho văn bản
box_image_2.setAlignment(Qt.AlignCenter)

# # Tên box chứa ảnh
# box_image_1_name = QLabel("Chữ ký 1", window)
# box_image_1_name.setGeometry(390+60+20+10, 110+155+60+10-30, 200, 30)
# box_image_1_name.setStyleSheet("color: red; font-size: 16px;")
#
# box_image_2_name = QLabel("Chữ ký 2", window)
# box_image_2_name.setGeometry(690+60+20+10, 110+155+60+10-30, 200, 30)
# box_image_2_name.setStyleSheet("color: red; font-size: 16px;")

# Ô kết quả
box_result = QLabel('', window)
box_result.setGeometry(460, 375, 440, 180)
box_result.setStyleSheet('background-color: white; border: 1px solid black; border-radius: 10px;')

result_label = QLabel("Result", window)
result_label.setGeometry(460, 385, 440, 30)
result_label.setStyleSheet("color: black; font-size: 16px; text-decoration: underline;")
result_label.setAlignment(Qt.AlignCenter)

# Kết quả
text_result = QLabel('', window)
text_result.setGeometry(460, 375, 440, 180)
# Đặt căn giữa văn bản theo chiều ngang và dọc
text_result.setAlignment(Qt.AlignCenter)

text_reliability = QLabel('', window)
text_reliability.setGeometry(460, 375, 440, 270)
text_reliability.setAlignment(Qt.AlignCenter)

window.setGeometry(100, 100, 1020, 600)
window.setStyleSheet(f"background-color: ${background_color};")
window.show()

sys.exit(app.exec_())
