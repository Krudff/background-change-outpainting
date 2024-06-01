import cv2
from PIL import Image
import mediapipe as mp
import numpy as np
from outpainting import *


def cv2_to_pil(cv2_image):
    """
    Converts a cv2 image (OpenCV image represented as a NumPy array) to a PIL.Image.Image object.

    Args:
        cv2_image (numpy.ndarray): OpenCV image represented as a NumPy array.

    Returns:
        PIL.Image.Image: Converted PIL image object.
    """

    # Convert color space from BGR (cv2) to RGB (PIL) if necessary
    if cv2_image.shape[2] == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2_image

    # Create a PIL image from the RGB NumPy array
    pil_image = Image.fromarray(rgb_image)
    return pil_image
def pil_to_cv2(pil_image):
    """
    Converts a PIL image to a OpenCV image (cv2).

    Args:
        pil_image (PIL.Image.Image): PIL image object.

    Returns:
        numpy.ndarray: OpenCV image represented as a NumPy array.
    """
    # Convert the PIL image to a NumPy array
    cv2_image = np.array(pil_image)

    # Convert color space from RGB (PIL) to BGR (cv2) if necessary
    if cv2_image.shape[2] == 3:
        bgr_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = cv2_image

    return bgr_image

def process_videos(video_files):
    all_concatenated_frames = []
    #Initialize the selfie segmentation model
    mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    #open video files
    caps = [cv2.VideoCapture(video_file) for video_file in video_files]
    middle = len(caps)//2
    _, bg_init = caps[middle].read()
    bg_pil = cv2_to_pil(bg_init).resize((512,512))
    bg_image_left = outpaint(bg_pil,middle,0)
    bg_image_right = outpaint(bg_image_left,len(caps)-middle-1,1)
    bg_image = bg_image_right
    #open background image
    image = pil_to_cv2(bg_image)
    height, width = image.shape[:2]
    
    #Calculate width for each video frame in concatenated image
    #can set a minimum width for each frame to ensure they are not too small
    min_frame_width = 160
    frame_width = max(width // len(video_files), min_frame_width)

    while True:
        frames = []
        masks = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                break

            #resize frame to calculated width while maintaining aspect ratio
            frame_height = int(frame.shape[0] * (frame_width / frame.shape[1]))
            frame = cv2.resize(frame, (frame_width, frame_height))

            #convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #process the frame
            results = mp_selfie_segmentation.process(frame_rgb)

            #create empty mask with same shape as frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            #post processing ey
            kernel = np.ones((5,5),np.uint8)
            mask = cv2.erode(mask, kernel, iterations = 1)
            mask = cv2.dilate(mask, kernel, iterations = 1)
            mask = cv2.GaussianBlur(mask, (61, 61), 0)

            #get condition where segmentation mask is 1 (i.e., person is present)
            condition = results.segmentation_mask > 0.9

            #apply condition to the frame
            if i == middle:
                mask[:] = 255
            else:
                mask[condition] = 255

            frames.append(frame)
            masks.append(mask)

        #ensure background image is large enough to accommodate the frames
        bg_image = cv2.resize(image, (frame_width * len(video_files), frame_height))

        concatenated_frame = None
        for i in range(len(frames)):
            #apply condition to the background image
            bg_image_resized = bg_image[:, frame_width*i:frame_width*(i+1)]
            bg_image_resized[masks[i] == 255] = frames[i][masks[i] == 255]

            #concat the frames
            if concatenated_frame is None:
                concatenated_frame = bg_image_resized
            else:
                concatenated_frame = np.concatenate((concatenated_frame, bg_image_resized), axis=1)

        #display the frame with the background changed
        try:
            all_concatenated_frames.append(concatenated_frame)
            cv2.imshow('Video with Background Changed', concatenated_frame)
        except cv2.error as e:
            print("Debugger: process finished!")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()
    return all_concatenated_frames

def save_video(frame_array, video_filename, fps=30.0, fourcc="mp4v"):#fps is according to video property (adjust if necessary)
    """
    Saves an array of NumPy arrays (representing video frames) to an mp4 video file at 30fps.

    Args:
        frame_array (list): List of NumPy arrays representing video frames.
        video_filename (str): Name of the output video file (including .mp4 extension).
        fps (float, optional): Frames per second for the video. Defaults to 30.0 (recommended for mp4).
    """

    # Get the frame dimensions from the first frame
    height, width, channels = frame_array[0].shape

    # Create a video writer object with mp4 codec (fourcc="mp4v")
    writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

    # Write each frame to the video
    for frame in frame_array:
        writer.write(frame)

    # Release the video writer
    writer.release()

    print(f"Video saved to: {video_filename}")

# Call the function with your video files and background image
frames = process_videos(['d4.mp4',"d3.mp4", 'd1.mp4',"d5.mp4"])
save_video(frames,"output_video.mp4")
#process_videos_v(['a5.mp4', 'a6.mp4', 'a7.mp4', 'a8.mp4'], 'bg3.jpg','output_mp3.mp4')
