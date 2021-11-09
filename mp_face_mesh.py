import cv2
import mediapipe as mp
from matplotlib import pyplot as plt


class color:
    # Color difine
    purple = (245,66,230)
    blue = (245,117,66)
    red = (0, 0, 255)
    green = (0, 255, 0)
    black = (0, 0, 0)


def img_face_mesh_get_irises(IMAGE_FILES):
    '''For static images'''
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # IMAGE_FILES = []
    keypoints = []
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                               refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
        
        annotated_image = image.copy()
        img_h, img_w  = annotated_image.shape[0], annotated_image.shape[1]
        print(f'w: {img_w}, h:{img_h}')
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)

            for data_point in face_landmarks.landmark:
                keypoints.append({
                         'X': data_point.x,
                         'Y': data_point.y,
                        #  'Z': data_point.z,
                        #  'Visibility': data_point.visibility,
                         })
            # print(f'len: {len(keypoints)}')
            # print(f'face_landmarks:\n {keypoints}')
            
            # # To look up specific point.
            # for i, j in enumerate(keypoints):
            #     # print('keypoints: ', i, j)
            #     # print(i, j['X']*img_w, j['Y']*img_h)
            #     if (j['X']*img_w) > 505 and (j['X']*img_w) <= 510:
            #         print(i, j['X']*img_w, j['Y']*img_h)
            #     else:
            #         pass
            
            # l_eye_up = 470, l_eye_down = 145, r_eye_up= 475, r_eye_down= 374 
            l_eye_up_x = int(keypoints[470].get('X')*img_w)
            l_eye_up_y = int(keypoints[470].get('Y')*img_h)

            l_eye_down_x = int(keypoints[145].get('X')*img_w)
            l_eye_down_y = int(keypoints[145].get('Y')*img_h) 

            r_eye_up_x = int(keypoints[475].get('X')*img_w)
            r_eye_up_y = int(keypoints[475].get('Y')*img_h) 

            r_eye_down_x = int(keypoints[374].get('X')*img_w)
            r_eye_down_y = int(keypoints[374].get('Y')*img_h)  

            num = 10
            line_thickness = 2
            cv2.line(annotated_image, (l_eye_up_x, l_eye_up_y-num), (l_eye_up_x, l_eye_up_y+num), color.blue, line_thickness)
            cv2.line(annotated_image, (l_eye_up_x-num, l_eye_up_y), (l_eye_up_x+num, l_eye_up_y), color.blue, line_thickness)

            cv2.line(annotated_image, (l_eye_down_x, l_eye_down_y-num), (l_eye_down_x, l_eye_down_y+num), color.green, line_thickness)
            cv2.line(annotated_image, (l_eye_down_x-num, l_eye_down_y), (l_eye_down_x+num, l_eye_down_y), color.green, line_thickness)

            cv2.line(annotated_image, (r_eye_up_x, r_eye_up_y-num), (r_eye_up_x, r_eye_up_y+num), color.blue, line_thickness)
            cv2.line(annotated_image, (r_eye_up_x-num, r_eye_up_y), (r_eye_up_x+num, r_eye_up_y), color.blue, line_thickness)

            cv2.line(annotated_image, (r_eye_down_x, r_eye_down_y-num), (r_eye_down_x, r_eye_down_y+num), color.green, line_thickness)
            cv2.line(annotated_image, (r_eye_down_x-num, r_eye_down_y), (r_eye_down_x+num, r_eye_down_y), color.green, line_thickness)

            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None, # drawing_spec
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # cv2.imwrite(IMAGE_FILES[0] + str(idx) + '_face_mesh_out.png', annotated_image)
        
        # BGR to RGB
        img_rgb = annotated_image[:,:,::-1]
        plt.imshow(img_rgb)
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Done!')


def img_mp_face_mesh(IMAGE_FILES):
    '''For static images'''
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # For static images:
    # IMAGE_FILES = []

    keypoints = []
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                               refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
        
        annotated_image = image.copy()
        img_h, img_w  = annotated_image.shape[0], annotated_image.shape[1]
        print(f'w: {img_w}, h:{img_h}')

        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)

            # for data_point in face_landmarks.landmark:
            #     keypoints.append({
            #              'X': data_point.x,
            #              'Y': data_point.y,
            #              'Z': data_point.z,
            #              'Visibility': data_point.visibility,
            #              })
            # print(f'len: {len(keypoints)}')
            # print(f'face_landmarks:\n {keypoints}')
            
            # draw face mesh with all face landmarks
            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # draw face mesh
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # draw face contour
            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # draw FACEMESH_IRISES
            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        cv2.imwrite(IMAGE_FILES[0] + str(idx) + '_face_mesh_out.png', annotated_image)
        
        # BGR to RGB
        img_rgb = annotated_image[:,:,::-1]
        plt.imshow(img_rgb)
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Done!')


def cap_mp_face_mesh(cap):
    '''For webcam input'''
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # draw face mesh with all face landmarks
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=drawing_spec,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    # draw face mesh
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    # draw face contour
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                    # draw eye
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()


def show_a_img(img):
    img_bgr = cv2.imread(img)
    h, w = img_bgr.shape[0], img_bgr.shape[1]
    print(f'h: {h}, w: {w}')

    # BGR to RGB
    # img_rgb = img_bgr[:,:,::-1]
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
    # cv2.line(img_rgb, (150, 0), (150, 100), color.red, 5)
    # cv2.line(img_rgb, (20,10), (100,10), color.blue, 2)

    # new_x = int(w/2)
    new_x = 265

    # cv2.line(img_bgr, (new_x, 0), (new_x, h), color.red, 6) # draw a line

    # BGR to RGB
    img_rgb = img_bgr[:,:,::-1]  

    plt.imshow(img_rgb)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_face_mesh_get_irises(IMAGE_FILES=['./people_face1.jpg'])
    # cap_mp_face_mesh(cap=cv2.VideoCapture(0))
    # show_a_img('./people_face.jpg')
