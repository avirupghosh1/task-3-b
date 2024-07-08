import cv2
import sys
import numpy as np
url = "http://192.168.0.113:8080/video"
cap = cv2.VideoCapture(url)
matrix_coefficients=np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
distortion_coefficients=np.array((-0.43948,0.18514,0,0))
if not cap.isOpened():
    print("Error: Unable to open the video stream")
    exit()

# Define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Select the ArUco dictionary
args = "DICT_6X6_1000"
def draw_axes(image, origin, axis_length):
    x_axis_end = (origin[0] + axis_length, origin[1])
    y_axis_end = (origin[0], origin[1] - axis_length)
    z_axis_end = (origin[0] - int(axis_length * 0.707), origin[1] + int(axis_length * 0.707))  

    # Draw the X axis in red
    cv2.arrowedLine(image, origin, x_axis_end, (0, 0, 255), 2)
    cv2.putText(image, 'X', (x_axis_end[0] + 10, x_axis_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the Y axis in green
    cv2.arrowedLine(image, origin, y_axis_end, (0, 255, 0), 2)
    cv2.putText(image, 'Y', (y_axis_end[0] - 20, y_axis_end[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the Z axis in blue
    cv2.arrowedLine(image, origin, z_axis_end, (255, 0, 0), 2)
    cv2.putText(image, 'Z', (z_axis_end[0] - 20, z_axis_end[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(args) is None:
        print(f"[INFO] ArUCo tag type '{args}' is not supported")
        sys.exit(0)

    # Load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    arucoDict =  cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args])
    arucoParams = cv2.aruco.DetectorParameters()
    
    detector=cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected_img_points = detector.detectMarkers(image)

    # Verify at least one ArUCo marker was detected0
    
    if len(corners) > 0:
        # Flatten the ArUCo IDs list
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            # Extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 0.02, matrix_coefficients,distortion_coefficients)
            cv2.drawFrameAxes(image, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01) 
            # Draw the bounding box of the ArUCo detection
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
           
            # Compute and draw the center (x, y)-coordinates of the ArUCo marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            # Draw the ArUCo marker ID on the image
            cv2.putText(image, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print(f"[INFO] ArUco marker ID: {markerID}")

    # Show the output image
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
