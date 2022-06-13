#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)


import cv2
import mediapipe as mp
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import cv2 as cv2_imshow
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
import smtplib,imghdr
from email.message import EmailMessage


def image_detect(orig_image):
    detected_person = 0
    box = []
    import os, json, cv2, random,sys,argparse
    from vision.ssd.config.fd_config import define_img_size
    define_img_size(640)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
    from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    label_path = "./models/voc-model-labels.txt"
    test_device = 'cpu:0'
    class_names = [name.strip() for name in open(label_path).readlines()]
    model_path = "models/pretrained/version-slim-320.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=1500, device=test_device)
    net.load(model_path)
    boxes, labels, probs = predictor.predict(orig_image, 1500/2 , 0.6)
    detected_person = len(probs)
    if len(probs) > 0:
        box = boxes[0]
    return detected_person,box

def makeMevector(img_url,mean):
    try:
        tmp_img   = img_url
        mean_img  = np.mean(tmp_img, axis=(0, 1))
        if not any(mean):
            mean = mean_img
                        
        hx = tmp_img.shape[0]
        wy = tmp_img.shape[1]
        if  hx > wy:
            half_size = int((hx-wy)/2.0)
            tmp_img  = cv2.copyMakeBorder(tmp_img, 0,0,half_size,(hx-wy-half_size),
                                                cv2.BORDER_CONSTANT,value= [0,0,0])
        if  wy > hx:
            half_size = int((wy-hx)/2.0)
            tmp_img  = cv2.copyMakeBorder(tmp_img,half_size,(wy-hx-half_size),0,0,
                                                cv2.BORDER_CONSTANT,value= [0,0,0])
                
        tmp_blp  = cv2.dnn.blobFromImage(tmp_img, 1, (224, 224), mean, False,crop = False)
        mdl_cnn.setInput(tmp_blp)
        tmp_vect = mdl_cnn.forward(name_lyr).reshape(1,-1)
        
        return tmp_vect
    except:
            print("!! HATA !! :: trg.txt ile resimler uyusmuyor... !!!")
#            sys.exit(0)



cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


 
from detectron2.utils.visualizer import ColorMode

mp_face_mesh = mp.solutions.face_mesh
j = 0 
ismask = 1
status = "status: Please put the Mask on well"

last_one = 0
mask_point = [0,2,11,12,13,14,15,16,17,18,19,20,37,38,39,40,41,42,43,57,60,61,62,72,73,74,76,77,78,
                80,81,82,83,84,85,86,87,88,89,90,91,92,94,95,96,97,98,99,106,125,141,146,
                164,165,167,178,179,180,181,182,183,184,185,186,191,194,200,201,202,204,
                206,211,216,241,242,250,267,268,269,270,271,272,273,287,290,291,292,302,303,304,305,
                306,307,308,310,311,312,313,314,315,316,317,318,319,320,321,322,324,325,326,327,328,335,354,
                370,375,391,393,402,403,404,405,406,407,408,409,410,415,418,421,
                422,424,426,431,458,460,461,462]

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  
im = cv2.VideoCapture(0)
while im.isOpened():
    status = "status: Please put the Mask on well"
    success, image = im.read()
    height,width,_ = image.shape
    liste = [[0 for x in range(width)] for y in range(height)]
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    outputs = predictor(image)  
    MASK_KONTROL = len(outputs["instances"].pred_boxes.tensor)

    if (MASK_KONTROL > 0):
        liste = outputs["instances"].pred_masks[0] 
        for m in range(1,MASK_KONTROL):
            liste = liste + outputs["instances"].pred_masks[m] 
        
    
    #            
    liste_length = len(mask_point)
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, 
                                min_tracking_confidence=0.6, 
                                min_detection_confidence=0.6) as face_mesh:
    
    
   #  fotoğraf mediapipe için hazırlama
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        # Draw the face detection annotations on the image.    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
          for multi_landmarks in results.multi_face_landmarks:
            for i in range(0, 467):
                point = multi_landmarks.landmark[i]
                x = int(point.x * width)
                y = int(point.y * height)
                for j in range(0, liste_length):
                    if (mask_point[j] == i):
                        if (liste[y][x] != True):
                            ismask = ismask -1
                       
        
        print(liste_length)
        if (ismask == 1):
            v = Visualizer(image[:, :, ::-1],scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            status = "excellent: Mask"
            cv2.imshow("Image",out.get_image()[:, :, ::-1])
        else:
            status = "status: Please put the Mask on well"
            if (ismask < -100):
                status = "status: No Mask "
                isperson,image_det = image_detect(image)
                if isperson > 0:
                    box = image_det
                    ractangle_image = image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                    proto    = "fcc_128.prototxt"
                    weights  = "fcc_128.caffemodel"
                    mean     = [91.4953, 103.8827, 131.0912]
                    mdl_cnn  = cv2.dnn.readNetFromCaffe(proto, weights)
                    name_lyr = 'feat_extract'
                    person = makeMevector(ractangle_image,mean)
                    from scipy import spatial
                    result_path = "./detected_img"
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    listdir_last = os.listdir("detected_img")
                    for file_path_last in listdir_last:
                        img_path_last = os.path.join("detected_img", file_path_last)
                        orig_image_last = cv2.imread(img_path_last)
                        data_person_last = makeMevector(orig_image_last,mean)
                        result = 1 - spatial.distance.cosine(person,data_person_last)
                        if result > 0.60:
                            last_one = 1
                    if last_one == 0:        
                        listdir = os.listdir("data")
                        for file_path in listdir:
                            img_path = os.path.join("data", file_path)
                            orig_image = cv2.imread(img_path)
                            data_person = makeMevector(orig_image,mean)
                            result = 1 - spatial.distance.cosine(person,data_person)
                            if result > 0.65:


                                alici_email_address = file_path.replace('.jpeg', '')
                                alici_email_address = file_path.replace('.jpg', '')
                                alici_email_address = file_path.replace('.png', '')
                                New_path = os.path.join("detected_img", file_path)
                                cv2.imwrite(New_path, ractangle_image) 
                                #msg = EmailMessage()
                                #msg['subject'] = 'Notification of violation'
                                #msg['From'] = ''
                                #msg['To'] = alici_email_address
                                #msg.set_content('You have violated the saftey measures by removing your mask. Please put on face mask.')
                                #with open('detected_img\\'+alici_email_address+'.png', 'rb') as f:
                                 #   file_data = f.read
                                 #   file_type = imghdr.what(f.name)
                                 #   file_name = f.name
                                #with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
                                 #   smtp.login('', '')
                                 #   smtp.send_message(msg)

                    last_one = 0  
            cv2.putText(image,status, (20,70) , cv2.FONT_HERSHEY_SIMPLEX , 1,(255, 0, 0),2, cv2.LINE_AA)
            ismask = 1
            cv2.imshow('Image', image)
    cv2.resizeWindow("Image", 800, 400)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
im.release()
cv2.destroyAllWindows()


