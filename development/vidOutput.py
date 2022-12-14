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
    if type(media) == "List" :
        makeVideo(media)
        return
    cv2.imwrite("output/" + "output" + ".jpg", media)
    cv2.imshow("Result", media)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
