from moviepy.editor import *
from pathlib import Path
import cv2
import numpy as np
  

# since moviepy.editor using RGB channels, while 
# openCV uses BGR, we need to need to shift things
def shiftChannels(images) :
    for i in range(len(images)) :
        images[i] = cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
    return images

def makeVideo(images):

    images = shiftChannels(images)
    img_clips = []

    fps = 24

    for i in images :
        slide = ImageClip(i,duration=1/fps)
        img_clips.append(slide)

    #concatenating slides
    video_slides = concatenate_videoclips(img_clips, method='compose', )
    #exporting final video
    video_slides.write_videofile("output_video.mp4", fps)

def makeMedia(media) :
    if ()





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