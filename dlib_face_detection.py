# pip install imutils
import dlib
import cv2
import imutils
import numpy as np

def img_face_detector1(img):
    # 縮小圖片
    img = imutils.resize(img, width=640)

    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()

    # 偵測人臉
    face_rects = detector(img, 0)

    # 取出所有偵測的結果
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

    # 顯示結果
    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_face_detector2(img):
    '''Useinf dlib face detector get all face features. Input img, output img.'''
    
    # convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # display_img(gray)

    detector = dlib.get_frontal_face_detector()

    # rects contains all the faces detected
    rects = detector(gray, 1)

    predictor = dlib.shape_predictor('68_face_landmarks.dat')

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        # print(f'type: {type(shape)}')
        shape = shape_to_np(shape)
        # print(f'shape: {shape}')

    for x, y in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    
    display_img(img)

    return img


def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def display_img(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cap_dlib_img_face_detector(cap):
    '''Using camera get face dection on real time.'''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('68_face_landmarks.dat')

    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]

    # cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    thresh = img.copy()

    cv2.namedWindow('image')
    kernel = np.ones((9, 9), np.uint8)

    def nothing(x):
        pass
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for rect in rects:

            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(mask, left, shape)
            mask = eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            threshold = cv2.getTrackbarPos('threshold', 'image')
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2) #1
            thresh = cv2.dilate(thresh, None, iterations=4) #2
            thresh = cv2.medianBlur(thresh, 3) #3
            thresh = cv2.bitwise_not(thresh)
            contouring(thresh[:, 0:mid], mid, img)
            contouring(thresh[:, mid:], mid, img, True)
            # for (x, y) in shape[36:48]:
            #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        # show the image with the face detections + facial landmarks
        cv2.imshow('eyes', img)
        cv2.imshow("image", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Image test.
    # img_face_detector1(img=cv2.imread('people.jpg'))
    # img_face_detector2(img=cv2.imread('people.jpg'))

    # Camera test.
    cap_dlib_img_face_detector(cap=cv2.VideoCapture(0))
