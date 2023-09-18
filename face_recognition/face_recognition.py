import os
import cv2
import numpy as np
import face_recognition

# Specify the directory containing the face images
path = 'gambar'
images = []  # Initialize an empty list to store images
names = []   # Initialize an empty list to store image names (without file extensions)
List = os.listdir(path)  # List all files in the specified directory

# Iterate through the files in the directory
for name in List:
    # Read each image file and append it to the 'images' list
    img = cv2.imread(f'{path}/{name}')
    images.append(img)
    # Extract the name without the file extension and store it in the 'names' list
    names.append(os.path.splitext(name)[0])

# Function to find face encodings for all images in the directory
def encode(images):
    encode_list = []  # Initialize an empty list to store face encodings
    # Iterate through images and their corresponding names
    for img, name in zip(images, names):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encode_list.append(encodeImg)
        print("Encoding for", name, "completed.")  # Print the name of the encoded image
    return encode_list

print("Encoding Faces ...")
encodeList = encode(images)
print("Encoding Completed.")

# Function to set up the camera using GStreamer
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Function for face recognition
def faceRecognize():
    # Initializing the camera
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        cv2.namedWindow("Face Recognition Window", cv2.WINDOW_AUTOSIZE)
        
        while True:
            return_key, image = video_capture.read()
            if not return_key:
                break

            # Reducing size of real-time image to 1/4th
            imgResize = cv2.resize(image, (0,0), None, 0.25, 0.25)
            imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)

            # Finding face in the current frame
            face = face_recognition.face_locations(imgResize)
            # Encode detected face
            encodeImg = face_recognition.face_encodings(imgResize, face)

            # Finding matches with existing images
            for encodecurr, loc in zip(encodeImg, face):
                match = face_recognition.compare_faces(encodeList, encodecurr)
                faceDist = face_recognition.face_distance(encodeList, encodecurr)
                
                # Lowest distance will be the best match
                index_BestMatch = np.argmin(faceDist)

                if match[index_BestMatch]:
                    name = names[index_BestMatch]
                else:
                    name = "unknown"

                print("Face Detected:", name)
                y1,x2,y2,x1 = loc
                # Retaining original image size for rectangle location
                y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),1)
                cv2.rectangle(image,(x1,y2-30),(x2,y2), (255,0,255), cv2.FILLED)
                cv2.putText(image, name, (x1+8, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255),2)

            cv2.imshow("Face Recognition Window", image)

            key = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if key == 27:
                break

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open Camera")

if __name__ == "__main__":
    faceRecognize()