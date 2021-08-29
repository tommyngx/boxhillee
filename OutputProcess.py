import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import cv2
import cv2
from google.colab.patches import cv2_imshow
from PIL import ImageColor
import colorsys
import random
import matplotlib.pyplot as plt
from PIL import Image
from YOLO4 import YOLOV41 # YOLOV4 Backbone
from LoadWeights import  WeightReader   # Load pre-trained weights from Darknet FrameWork
from PreprocessImage import * #load_image_pixels  # Load image


input_w, input_h = 416, 416
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
 
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
 
        return self.label
 
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
 
        return self.score
 
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
 
def decode_netout(netout, anchors, obj_thresh, net_h, net_w, nb_box, scales_x_y):
    grid_h, grid_w = netout.shape[:2]  
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5 # 5 = bx,by,bh,bw,pc

    # print("grid_h,grid_w: ",grid_h,grid_w)   
    # print("nb class: ",nb_class)   
    
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2]) # x, y
    netout[..., :2] = netout[..., :2]*scales_x_y - 0.5*(scales_x_y - 1.0) # scale x, y

    netout[..., 4:] = _sigmoid(netout[..., 4:]) # objectness + classes probabilities

    for i in range(grid_h*grid_w):

        row = i / grid_w
        col = i % grid_w
        
        
        for b in range(nb_box):
            # 4th element is objectness
            objectness = netout[int(row)][int(col)][b][4]
            # print("objectness: ",objectness)                

            if(objectness > obj_thresh):
                # print("objectness: ",objectness)                
            
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height 
            
                # last elements are class probabilities
                classes = objectness*netout[int(row)][col][b][5:]
                classes *= classes > obj_thresh
                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)           
                boxes.append(box)
    return boxes

# Compute the Yolo layers
def yolo_boxes(yhat):
    obj_thresh = 0.6
    anchors = [ [12, 16, 19, 36, 40, 28],[36, 75, 76, 55, 72, 146],[142, 110, 192, 243, 459, 401]]
    scales_x_y = [1.2, 1.1, 1.05]
    boxes = list()

    for i in range(len(anchors)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], obj_thresh, input_h, input_w, 3, scales_x_y[i])
    # print("nb boxes detected; ",len(boxes))
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
            y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    return boxes

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3
 
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    # print(w1*h1)
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0



def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] ), int(x[1] ), int(x[2] )), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh, colors):
    v_boxes, v_labels, v_scores, v_colors = list(), list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):

            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                v_colors.append(colors[i])
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores, v_colors



# Vẽ
def draw_boxes(filename, v_boxes, v_labels, v_scores, v_colors):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color=v_colors[i])
        ax.add_patch(rect)
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='red')
    pyplot.show()

def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c
def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))
def draw_boxes2(filename, v_boxes, v_labels, v_scores, v_colors):
    v_colors=['#F657C6','#9BEC1C','#DE1F55','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        # For bounding box
        # For the text background
        color2 = v_colors[i]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 1)
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)

        # Prints the text. 
        img = cv2.rectangle(img, (x1, y1-25), (x1 + w, y1), color2, -1)
        text_color=v_colors[i]
        text_color2 = ImageColor.getcolor(text_color, "RGB")
        text_color2 = complement(*text_color2)
        img = cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX,0.77, text_color2, 1,cv2.LINE_AA)
        # For printing text
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imwrite("result.jpg",img)
    #cv2_imshow(img)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def draw_boxes3(filename, v_boxes, v_labels, v_scores, v_colors):
    v_colors=['#F657C6','#9BEC1C','#DE1F55','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    #print(v_boxes[1])
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        #print(i2)
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        # For bounding box
        # For the text background
        color2 = v_colors[i2]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 1) #
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)

        # Prints the text. 
        img = cv2.rectangle(img, (x1, y1-25), (x1 + w, y1), color2, -1)
        text_color=v_colors[i2]
        text_color2 = ImageColor.getcolor(text_color, "RGB")
        text_color2 = complement(*text_color2)
        img = cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX,0.77, text_color2, 1,cv2.LINE_AA)
        # For printing text
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imwrite("result.jpg",img)
    #cv2_imshow(img)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def crop_boxes4(filename, v_boxes, v_labels, v_scores, v_colors):
    v_colors=['#F657C6','#9BEC1C','#DE1F55','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    labelshow=[]
    k=0
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          labelshow.append("%s:%.0f" % (v_labels[i], v_scores[i]) + "%")
          crop = img[y1:y2, x1:x2]
          cv2.imwrite("crop_{}.jpg".format(k), crop)
          k=k+1
        if i2==0:
          #print("Đốt Xương {}".format(i))
          labelshow.append("%s:%.0f" % (v_labels[i], v_scores[i]) + "%")
          crop2 = img[y1:y2, x1:x2]
          cv2.imwrite("crop_{}.jpg".format(k), crop2)
          k=k+1

    fig = plt.figure(figsize=(25, 12))
    columns = 4
    rows = 4
    for i in range(1, len(labelshow)+1):
        img = cv2.imread("crop_{}.jpg".format(i-1))
        i2=i-1
        plt.rc('font', size=15) 
        if labelshow[i2][0]=="V":
          fig.add_subplot(rows, columns, i).set_title('{}'.format(labelshow[i2]), color='r')
        elif labelshow[i2][0]=="A":
          fig.add_subplot(rows, columns, i).set_title('{}'.format(labelshow[i2]), color='g')
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def crop_vert_calibrate(filename, v_boxes, v_labels, v_scores, percentreduce):
    v_colors=['#F657C6','#9BEC1C','#DE1F55','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    labelshow=[]
    k=0
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          labelshow.append("%s:%.0f" % (v_labels[i], v_scores[i]) + "%")
          y1,y2 = int(percentreduce*y1), int(y2)
          crop = img[y1:y2, x1:x2]
          cv2.imwrite("crop_{}.jpg".format(k), crop)
          k=k+1
        if i2==0:
          #print("Đốt Xương {}".format(i))
          labelshow.append("%s:%.0f" % (v_labels[i], v_scores[i]) + "%")
          y1,y2 = int(percentreduce*y1), int(y2)
          crop2 = img[y1:y2, x1:x2]
          cv2.imwrite("crop_{}.jpg".format(k), crop2)
          k=k+1

    fig = plt.figure(figsize=(25, 12))
    columns = 4
    rows = 4
    for i in range(1, len(labelshow)+1):
        img = cv2.imread("crop_{}.jpg".format(i-1))
        i2=i-1
        plt.rc('font', size=15) 
        if labelshow[i2][0]=="V":
          fig.add_subplot(rows, columns, i).set_title('{}'.format(labelshow[i2]), color='r')
        elif labelshow[i2][0]=="A":
          fig.add_subplot(rows, columns, i).set_title('{}'.format(labelshow[i2]), color='g')
        plt.imshow(img)
        plt.axis('off')
    plt.show()



def show_vertebral(link, size_reduce):
  labels =['Vertebra','Abnormal','Spine','Sacrum']

  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.5
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes3('temp.jpg', v_boxes_rs, v_labels, v_scores, v_colors)
  crop_boxes4('temp.jpg', v_boxes_rs, v_labels, v_scores, v_colors)

def show_vertebralX(link, size_reduce):
  labels =['Vertebra','Abnormal','Spine','Sacrum']

  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.5
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes33('temp.jpg', v_boxes_rs, v_labels, v_scores, v_colors)
  crop_boxes4('temp.jpg', v_boxes_rs, v_labels, v_scores, v_colors)

def show_vertebral_ori(link, size_reduce):
  labels =['Vertebra','Abnormal','Spine','Sacrum']

  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.5
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes33('somepic.jpg', v_boxes_rs, v_labels, v_scores, v_colors)
  crop_boxes4('somepic.jpg', v_boxes_rs, v_labels, v_scores, v_colors)

def show_vertebral_calibrate(link, size_reduce, percentreduce):
  labels =['Vertebra','Abnormal','Spine','Sacrum']

  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.8
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes_calibrate('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce)
  crop_vert_calibrate('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce)

def show_vertebral_calibrate2(link, size_reduce, percentreduce,percentreuduce2):
  labels =['Vertebra','Abnormal','Spine','Sacrum']

  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.8
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes_calibrate2('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
  crop_vert_calibrate2('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)

def draw_boxes_calibrate(filename, v_boxes, v_labels, v_scores, percentreduce):
    v_colors=['#F657C6','#9BEC1C','#00B2FF','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    #print(v_boxes[1])
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        #print(i2)
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          y1,y2 = int(percentreduce*y1), int(y2)
        if i2==0:
          #print("Đốt Xương {}".format(i))
          y1,y2 = int(percentreduce*y1), int(y2)
        # For bounding box
        # For the text background
        color2 = v_colors[i2]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 1)
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)

        # Prints the text. 
        img = cv2.rectangle(img, (x1, y1-25), (x1 + w, y1), color2, -1)
        text_color=v_colors[i2]
        text_color2 = ImageColor.getcolor(text_color, "RGB")
        text_color2 = complement(*text_color2)
        img = cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX,0.77, text_color2, 1,cv2.LINE_AA)
        # For printing text
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imwrite("result.jpg",img)
    #cv2_imshow(img)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def draw_boxes33(filename, v_boxes, v_labels, v_scores, v_colors):
    v_colors=['#F657C6','#9BEC1C','#00B2FF','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    #print(v_boxes[1])
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        #print(i2)
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        # For bounding box
        # For the text background
        color2 = v_colors[i2]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 3) #
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)

        # Prints the text. 
        img = cv2.rectangle(img, (x1, y1-50), (x1 + w, y1), color2, -1)
        text_color="#000000"#v_colors[i2]
        text_color2 = ImageColor.getcolor(text_color, "RGB")
        #text_color2 = complement(*text_color2)
        img = cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX,1.77, text_color2, 2,cv2.LINE_AA)

    cv2.imwrite("result.jpg",img)
    #cv2_imshow(img)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def draw_boxes_calibrate2(filename, v_boxes, v_labels, v_scores, percentreduce, percentreuduce2):
    v_colors=['#F657C6','#9BEC1C','#00B2FF','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    #print(v_boxes[1])
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        #print(i2)
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)
        if i2==0:
          #print("Đốt Xương {}".format(i))
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)
        # For bounding box
        # For the text background
        color2 = v_colors[i2]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 3)
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)

        # Prints the text. 
        img = cv2.rectangle(img, (x1, y1-50), (x1 + w, y1), color2, -1)
        text_color="#000000"#v_colors[i2]
        text_color2 = ImageColor.getcolor(text_color, "RGB")
        text_color2 = complement(*text_color2)
        img = cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX,1.77, text_color2, 2,cv2.LINE_AA)
        # For printing text
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.imwrite("result.jpg",img)
    #cv2_imshow(img)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def crop_vert_calibrate2(filename, v_boxes, v_labels, v_scores, percentreduce, percentreuduce2):
    v_colors=['#F657C6','#9BEC1C','#DE1F55','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    labelshow=[]
    k=0
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          labelshow.append("%s:%.0f" % (v_labels[i], v_scores[i]) + "%")
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)
          crop = img[y1:y2, x1:x2]
          cv2.imwrite("crop_{}.jpg".format(k), crop)
          k=k+1
        if i2==0:
          #print("Đốt Xương {}".format(i))
          labelshow.append("%s:%.0f" % (v_labels[i], v_scores[i]) + "%")
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)
          crop2 = img[y1:y2, x1:x2]
          cv2.imwrite("crop_{}.jpg".format(k), crop2)
          k=k+1

    fig = plt.figure(figsize=(25, 12))
    columns = 4
    rows = 4
    for i in range(1, len(labelshow)+1):
        img = cv2.imread("crop_{}.jpg".format(i-1))
        i2=i-1
        plt.rc('font', size=15) 
        if labelshow[i2][0]=="V":
          fig.add_subplot(rows, columns, i).set_title('{}'.format(labelshow[i2]), color='r')
        elif labelshow[i2][0]=="A":
          fig.add_subplot(rows, columns, i).set_title('{}'.format(labelshow[i2]), color='g')
        plt.imshow(img)
        plt.axis('off')
    plt.show()

