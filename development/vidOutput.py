from moviepy.editor import *
from pathlib import Path
import cv2
import numpy as np
  

def makeVideo(images):
    img_clips = []

    fps = 24

    for i in images :
        slide = ImageClip(i,duration=1/fps)
        img_clips.append(slide)

    #concatenating slides
    video_slides = concatenate_videoclips(img_clips, method='compose', )
    #exporting final video
    video_slides.write_videofile("output_video.mp4", fps)



# # # creating an array using np.full 
# # # 255 is code for white color
# array_created = []
# array_created.append(np.full((500, 500, 3),
#                         255, dtype = np.uint8))
# img_clips = []
# array_created.append(np.full((500, 500, 3),
#                     255, dtype = np.uint8))

# print(type(array_created[0]))
# # # displaying the image

# img_clips = []

# for i in array_created :
#     slide = ImageClip(i,duration=2)
#     img_clips.append(slide)


# #concatenating slides
# video_slides = concatenate_videoclips(img_clips, method='compose')
# #exporting final video
# video_slides.write_videofile("output_video.mp4", fps=24)