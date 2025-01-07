import cv2
import os

input_directory = './'
output_video = 'video.mp4'
frame_rate = 24

image_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]
image_files.sort(key=lambda x: int(x[4:-4]))

image_path = os.path.join(input_directory, image_files[0])
img = cv2.imread(image_path)
height, width, _ = img.shape
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

for image_file in image_files:
    image_path = os.path.join(input_directory, image_file)
    img = cv2.imread(image_path)
    video.write(img)

cv2.destroyAllWindows()
video.release()