from tokenize import Number
from numpy import testing
from numpy.lib.type_check import imag
import pygame,sys
from pygame import image
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
from tensorflow.python.keras.backend import constant
image_cnt=1
PREDICT=True
WINDOWSIZEX=640
WINDOWSIZEY=480
BOUNDARYINC=5
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)
IMAGESAVE=False
isWriting=False
Model=load_model("bestmodel.h5")
LABELS={0:"Zero",1:"One",
        2:"Two",3:"Three",
        4:"Four",5:"Five",
        6:"Six",7:"Seven",
        8:"Eight",9:"Nine"
        }
pygame.init()
FONT=pygame.font.Font("OpenSans-BoldItalic.ttf",18)
DISPLAYSURF=pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))
WHIITE_INT=DISPLAYSURF.map_rgb(WHITE)
pygame.display.set_caption("Digital Board")
number_xcord=[]
number_ycord=[]
MODEL = load_model("bestmodel.h5")
WHITE_INT = DISPLAYSURF.map_rgb(WHITE)
while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==MOUSEMOTION and isWriting:
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAYSURF,WHITE,(xcord,ycord),4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type==MOUSEBUTTONDOWN:
            isWriting=True

        if event.type==MOUSEBUTTONUP:
            isWriting=False
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord)
            rect_min_x,rect_max_x=max(number_xcord[0]-BOUNDARYINC,0),min(WINDOWSIZEX,number_xcord[-1]+BOUNDARYINC)
            rect_min_y,rect_max_y=max(number_ycord[0]-BOUNDARYINC,0),min(number_ycord[-1]+BOUNDARYINC,WINDOWSIZEY)
            number_xcord=[]
            number_ycord=[]
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            if IMAGESAVE:
                cv2.imwrite("image-{%d}.png" % image_cnt, img_arr)
                image_cnt+=1
            if PREDICT:
                image=cv2.resize(img_arr,(28,28))
                image=np.pad(image,(10,10),'constant',constant_values=0)
                image = cv2.resize(image, (28, 28))/WHITE_INT
                label=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))]).title()
                textSurfaceObj = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurfaceObj.get_rect()
                textRectObj.left,textRectObj.bottom=rect_min_x,rect_max_y
                DISPLAYSURF.blit(textSurfaceObj, textRectObj)
            pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y,rect_max_x - rect_min_x, rect_max_y - rect_min_y), 3)    
        if event.type==KEYDOWN:
            if event.unicode=="n":
                DISPLAYSURF.fill(BLACK)
    pygame.display.update()