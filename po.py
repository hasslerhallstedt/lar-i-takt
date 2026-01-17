################Här är filens namn!###############
filnamn='nyfil_gang-1.csv'
##################################################
textskala=1.5*2#textskala
r1=int(1.5*70)##cirkelnsradie
r2=int(3*70) ##klappradie
#cd /Users/matticervin/Desktop/Martins\ saker/Från\ Max\ 250314
#python3 po2.py
import cv2
import mediapipe as mp
import math
import numpy as np
import threading
import os
import pandas as pd
#from playsound import playsound
import pygame
pygame.mixer.init()
pygame.init()

pf="pop.wav"
#pf2="C:/Users/ma8107th/Work Folders/Documents/pose/beat.wav"
pf3="Back on track.wav"

gs1=pygame.mixer.Sound(pf)
channel = pygame.mixer.Channel(1)  # Use channel 1
#gs2=pygame.mixer.Sound(pf2)
channel2 = pygame.mixer.Channel(2)

gs3=pygame.mixer.Sound(pf3)
channel3 = pygame.mixer.Channel(2)  # Use channel 1


df=pd.read_csv(filnamn,sep=';',encoding='utf-8')
print(df.head())
df['tid']=df['tid'].astype('float')
#print(df['merge_ord'])
#df['merge_plats'] = pd.concat([df['plats1'],df['plats2'],df['plats3']])
#os.exit()
# Check if the sound is already playing on this channel
#while channel.get_busy():

  #  elapsed_time = pygame.time.get_ticks() - start_time
   # print(f"Sound playing for: {elapsed_time / 1000:.2f} seconds")
if not channel2.get_busy():
             		    channel2.play(gs3)
             		    start_time3 = pygame.time.get_ticks()
#pygame.init()
#pygame.display.set_caption("OpenCV camera stream on Pygame")
#screen = pygame.display.set_mode([1280,720])
#mixer.init()


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
#cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
# font
def distance(SL,SR):
     return math.sqrt((SL[0]-SR[0])**2+(SL[1]-SR[1])**2)
import math
 
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
 

font = cv2.FONT_HERSHEY_DUPLEX

# org
org = (50, 50)
#my_class = MyClass()
# fontScale
fontScale = 1
 
# Blue color in BGR
color = (255, 0, 0)
color1 = (0, 0, 0)#(255, 255, 0)
color2 = (255, 255, 255)
# Line thickness of 2 px
thickness = 2
tid=df['tid'].values
orden=df['ord'].values
plats=df['plats'].values
ratt=df['ratt'].values
print('start')
import matplotlib.pyplot as plt
#plt.scatter(tid,orden)
#plt.show()
score=0
pointl=[]
pointevs=[]
indr=np.where(ratt==1)[0]

while cap.isOpened():
    # read frame
    _, frame = cap.read()
    frame=cv2.flip(frame, 1)
    frame1=frame.copy()
    elapsed_time = (pygame.time.get_ticks() - start_time3)/1000 ##+20
    dif=(tid-elapsed_time)
  #  print(elapsed_time)
   # print(dif)
    pm=False
    idd=np.where((dif<0)&(dif>-1.4))###hur länge det ska visas
    dif2=dif[indr]
    
    idd2=indr[np.where((dif2<1.5)&(dif2>-1.5))[0]]###tid klapp
    if len(idd2)>0:
       print(plats[idd2])
       pm=True
    
        
    try:
         # resize the frame for portrait ssideo
         # frame = cv2.resize(frame, (350, 600))
         # convert to RGB

         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
         # process the frame for pose detection
         pose_results = pose.process(frame_rgb)
         # print(pose_results.pose_landmarks)
         if not True:#not pm:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
         # draw skeleton on the frame
        # print(pose_results.pose_landmarks)
         mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
         # display the framsssssss
         h,w,d=frame_rgb.shape
        # w,h=frame.size()
         center_coordinates2=(int(w/5),int(h/4))#150)##vänster
         center_coordinates1=(int(w/2),int(h/2))#150)##mitten
         center_coordinates3=(int(w-w/5),int(h/4))#150)##höger
         po1=(0,0)
         wd=int(w/2-w/5)
         hd=int(h/2-h/4)
         po2=(wd,-hd)
         po3=(-wd,-hd)
         dic={}
         dic['mitten']=po1
         dic['Vhörn']=po3
         dic['Hhörn']=po2
         rp=[0,0,0]
       #  r1=int(1.5*70)
        # center_coordinates1=po1#(int(w/5),int(h/4))#150)##vänster
       #  center_coordinates2=po2#(int(w/2),int(h/2))#150)##mitten
      #   center_coordinates3=po3#(int(w-w/5),int(h/4))#150)##höger
         alpha=0.7
       #  cv2.circle(frame, po1,r1, (255,55,255), -1)
       #  cv2.circle(frame, po2, r1, (255,55,255), -1)
      #   cv2.circle(frame, po3, r1, (255,55,255), -1)
         cv2.circle(frame, center_coordinates1,r1, (255,255,255), -1)
         cv2.circle(frame, center_coordinates2, r1, (255,255,255), -1)
         cv2.circle(frame, center_coordinates3, r1, (255,255,255), -1)
         frame= cv2.addWeighted(frame, alpha, frame1, 1 - alpha, 0)
         pt=False
         if len(idd[0])==2:
             #  print(ratt[idd])
               pr=idd[0][np.where(ratt[idd[0]]==1)[0]]
               ind2=np.where(['mitten','Vhörn','Hhörn']==plats[pr])[0]
              # print('ind2;',ind2)
               
              # print('####')
               pt=True
         for o in idd[0]:
               #  print('##',o)
                 ordd=orden[o]     
                 pl=plats[o]
              #   print('##',ordd,pl)
                 if 'ä' in str(ordd):
                     ordd=ordd.replace('ä','a')
                 if len(str(pl))<4:
                     pl='mitten'
                          
                 try:
                     textsize = cv2.getTextSize(str(ordd), font, fontScale*textskala, thickness)[0]
                 #    print(textsize)                     
                     textX = (w - textsize[0]) / 2+dic[pl][0]
                     textY = (h + textsize[1]) / 2+dic[pl][1]
                #     print((textX,textY))
                    # textsize = cv2.getTextSize(str(ordd), font, 1, 2)[0]
                     cv2.putText(frame, str(ordd), (int(textX),int(textY)), font, 
					   fontScale*textskala, color1, thickness,cv2.LINE_AA)
                 except ValueError:
                     pass

        # print('x')
         landmarks=[]
         for landmark in pose_results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * w), int(landmark.y * h),
                                  (landmark.z * w)))
      #   print(landmarks)
         HL=(landmarks[20][0],landmarks[20][1])
         HR=(landmarks[19][0],landmarks[19][1])
         SL=(landmarks[12][0],landmarks[12][1])
         SR=(landmarks[11][0],landmarks[11][1])
         DRL=math.sqrt((HL[0]-HR[0])**2+(HL[1]-HR[1])**2)
         SRL=math.sqrt((SL[0]-SR[0])**2+(SL[1]-SR[1])**2)
         org=(int((HL[0]+HR[0])/2),int((HL[1]+HR[1])/2))
         tn=pygame.time.get_ticks()-start_time3
         tn2=tid-tn
         locs=np.where((tn2>-1)&(tn2<0))
         if len(locs[0])>0:
            
             texts=orden[locs]
            # print(texts)
         
       #  cv2.putText(frame, str(texts), (w/2,h/2), font, 
		         #  fontScale, color, thickness,cv2.LINE_AA)

         try:
             an1=getAngle((landmarks[12][0],landmarks[12][1]), (landmarks[14][0],landmarks[14][1]), (landmarks[16][0],landmarks[16][1]))
             if abs(an1)>180:
                  an1=abs(an1)-180
          #   cv2.putText(frame, str(round(an1,3)), (landmarks[14][0],landmarks[14][1]), font, 
		         #  fontScale, color, thickness,cv2.LINE_AA)
         except:
             pass
       #  try:
           #  cv2.putText(frame, str(round(SRL/DRL,3)), org, font, 
		   #        fontScale, color, thickness,cv2.LINE_AA)
       #  except:
        #     pass
         try:
             if True:#SRL/DRL>3.7:
             	#cv2.circle(frame, org, 30, (255,255,255), -1)
             	#klapp=True
             	dis1=[distance(a,HL) for a in [center_coordinates1,center_coordinates2,center_coordinates3]]
             	dis2=[distance(a,HR) for a in [center_coordinates1,center_coordinates2,center_coordinates3]]
             #	print(dis)
             	dp1=np.argmin(dis2)
             	dp2=np.argmin(dis1)
             	#print(dp1,dp2)
             	if dp1==dp2 and dis1[dp1]<r2 and dis2[dp2]<r2:
                    

             		#if not channel.get_busy():
             		  #  channel.play(gs1)
             		print('KLAPP')
             		ll=dp1
                        
             		cv2.circle(frame, [center_coordinates1,center_coordinates2,center_coordinates3][dp1], 10, (255,0,255), -1)
             		if len(idd2)>0:
                            print(plats[idd2])
                            if ['mitten','Vhörn','Hhörn'][ll]==plats[idd2]:
                              pr2=idd2
                          
                              if pr2 not in pointl:
                                      score+=1
                                      pointl.append(pr2)
                                 
             		if False:#pt:
                           if ll==ind2:
                              if pr not in pointl:
                                      score+=1
                                   #   pointl.append(pr)
                              print('SCORE!')
                      

         except:# ArithmeticError:
                 pass
         cv2.circle(frame, HR, 40, (0,255,0), 2)
         cv2.circle(frame, HL, 40, (0,0,255), 2)

         cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
         ts='Score: '+str(score)
         textsize = cv2.getTextSize(ts, font, fontScale, thickness)[0]
       #  print(textsize)                     
         textX = (w - textsize[0]) / 2#+w/2#dic[pl][0]
         textY = (h + textsize[1]) / 2+int(h/3.5)#dic[pl][1]
         cv2.putText(frame, ts, (int(textX),int(textY)), font, 
					   fontScale, color2, thickness,cv2.LINE_AA)
         cv2.imshow('frame', frame)
    except:# ValueError:
         pass
    if cv2.waitKey(1) == ord('q'):
          break
          
cap.release()
cv2.destroyAllWindows()
      #   print(tn2)		§
