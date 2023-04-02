import cv2
import numpy as np

ROW = 10
COL = 17
CROP_BOUDARY = 4


def classify(input_img):
    min = 999
    index = -1
    for i in range(1, 10):
        # print(i)
        num_img = cv2.imread(f'images/{i}.jpg', cv2.IMREAD_GRAYSCALE)
        diff = cv2.bitwise_xor(input_img, num_img)
        score = cv2.countNonZero(diff)
        # print('이미지 유사도:', score)
        if min > score:
            index = i
            min = score
    if index == -1:
        raise
    return index


filename = 'images/apple_game_image.png'
img = cv2.imread(filename)

lower_bound = (0, 0, 100)
upper_bound = (90, 90, 255)
imthres = cv2.inRange(img, lower_bound, upper_bound)

cnts, hierarchy = cv2.findContours(
    imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

apple_list = []
for i in reversed(range(len(cnts))):
    x, y, w, h = cv2.boundingRect(cnts[i])
    ROI = img[y+CROP_BOUDARY:y+h-CROP_BOUDARY, x+CROP_BOUDARY:x+w-CROP_BOUDARY]
    lower_bound = (210, 210, 210)
    upper_bound = (255, 255, 255)
    imthres = cv2.inRange(ROI, lower_bound, upper_bound)
    imthres = 255 - imthres
    apple_list.append(classify(imthres))

apple_array = np.reshape(apple_list, (ROW, COL))
print(apple_array)

cv2.imshow('CHAIN_APPROX_NONE', img)
cv2.waitKey()
