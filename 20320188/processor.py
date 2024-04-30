import cv2
import imutils
import numpy as np



def extract_key_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)  # Capture the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    interval = total_frames // num_frames  # Calculate the interval
    key_frames = []

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Get the position of the frame
        ret, frame = cap.read() # Capture the frame
        if ret:
            key_frames.append(frame)
        if len(key_frames) >= num_frames:
            break

    cap.release()  # Free up system resources
    return key_frames

def bilateral_filter(frames, d, sigmaColor, sigmaSpace):
    filtered_frames = []
    for frame in frames:
        filtered_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)  # apply bilateral filters
        filtered_frames.append(filtered_frame)
    return filtered_frames

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (centralX, centralY) = (w // 2, h // 2)  # Get the central point
    M = cv2.getRotationMatrix2D((centralX, centralY), angle, 1.0)  # Rotation Matrix Creation

    # Canvas Size Calculation
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    newW = int((h * sin) + (w * cos))
    newH = int((h * cos) + (w * sin))

    # Adjustment of Rotation Matrix for Translation
    M[0, 2] += (newW / 2) - centralX
    M[1, 2] += (newH / 2) - centralY
    rotated_frame = cv2.warpAffine(image, M, (newW, newH))  # Applies the affine transformation defined by matrix M to the original image
    return rotated_frame

def image_stitching(key_frames):
    stitcher = cv2.Stitcher_create() # Create a stitch object
    error, stitched_img = stitcher.stitch(key_frames)  # Stitch together a series of key frames

    return stitched_img

def post_process(stitched_img):
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))  # Add a black boarder

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # Binarized the grayscale

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Create a kernel
    closing = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)  # Operating a morphological closing operation

    # cv2.imshow('closing', closing)  # Show the picture after morphological closing operation
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours of closing image
    contours = imutils.grab_contours(contours)
    area = max(contours, key=cv2.contourArea)  # Find the main image area

    mask = np.zeros(closing.shape, dtype="uint8")  # Create a Blank Mask
    x, y, w, h = cv2.boundingRect(area)  # Find Bounding Rectangle
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # Draw Rectangle on the Mask

    # Copy the mask
    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)  # Perform the erode operation
        sub = cv2.subtract(minRectangle, closing)  # Calculate the difference

    # cv2.imshow("minRectangle", minRectangle)  # Show the minimal rectangle
    # cv2.waitKey(0)

    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours of minRectangle
    contours = imutils.grab_contours(contours)
    area = max(contours, key=cv2.contourArea)  # Find the main image area

    x, y, w, h = cv2.boundingRect(area)  # Calculate the smallest rectangle
    panorama = stitched_img[y:y + h, x:x + w]  # Crop the image

    return panorama


def process_horizontal_video(video_path, num_frames, blur_intensity):
    key_frames = extract_key_frames(video_path, num_frames)

    if blur_intensity == " No Blur":
        filtered_key_frames = bilateral_filter(key_frames, 0, 0, 0)
    # elif blur_intensity == " Less Blur":
        # filtered_key_frames = bilateral_filter(key_frames, 5, 50, 50)
    elif blur_intensity == " Moderate Blur":
        filtered_key_frames = bilateral_filter(key_frames, 9, 75, 75)
    # elif blur_intensity == " More Blur":
        # filtered_key_frames = bilateral_filter(key_frames, 15, 150, 150)

    panorama = image_stitching(filtered_key_frames)
    if panorama is not None:
        panorama = post_process(panorama)
        cv2.imwrite("Panorama.png", panorama)
        cv2.imshow('Panorama', panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_vertical_video(video_path, num_frames, blur_intensity):
    key_frames = extract_key_frames(video_path, num_frames)

    if blur_intensity == " No Blur":
        filtered_key_frames = bilateral_filter(key_frames, 0, 0, 0)
    # elif blur_intensity == " Less Blur":
        # filtered_key_frames = bilateral_filter(key_frames, 5, 50, 50)
    elif blur_intensity == " Moderate Blur":
        filtered_key_frames = bilateral_filter(key_frames, 9, 75, 75)
    # elif blur_intensity == " More Blur":
        # filtered_key_frames = bilateral_filter(key_frames, 15, 150, 150)

    key_frames_rotated = [rotate_image(frame, 90) for frame in filtered_key_frames]
    panorama = image_stitching(key_frames_rotated)
    if panorama is not None:
        panorama = post_process(panorama)
        panorama = rotate_image(panorama, -90)
        cv2.imwrite("Panorama.png", panorama)
        cv2.imshow('Panorama', panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


'''
# It was Main function when the UI hadn't been designed
video_path = '1.mp4'
key_frames = extract_key_frames(video_path, 30)
#key_frames_rotated = [rotate_image(frame, 90) for frame in key_frames]

#filtered_key_frames = bilateral_filter(key_frames, 0, 0, 0)
panorama = image_stitching(filtered_key_frames)
#panorama = image_stitching(key_frames_rotated)

if panorama is not None:
    panorama = post_process(panorama)
    #panorama = rotate_image(panorama, -90)
    cv2.imwrite("Panorama.png", panorama)
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''