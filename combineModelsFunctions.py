# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import easyocr
import os
from pathlib import Path
import omegaconf
import functions
import importlib
import re
import difflib
import time
importlib.reload(functions)

import cv2
reader = easyocr.Reader(['tr'], gpu=True)
model_char_CNN_ResNet='ResNet_character_recognition.txt'
model_weight_CNN_ResNet="ResNet50_License_character_recognition2.h5"
model_class_CNN_ResNet='Resnet50_License_chars.npy'
model_CNN,labels_CNN=functions.load_train_model(model_char_CNN_ResNet,model_weight_CNN_ResNet,model_class_CNN_ResNet)
wpod_net_path = "wpod-net.json"
wpod_net = functions.load_model(wpod_net_path)
eklenen_plates = set()
yolo_and_CNN=[]
predicted_plate_CNN=[]
start_time = time.time()
deneme=""
deneme_arr=[]
def ocr_image(img,coordinates):
    x,y,w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]),int(coordinates[3])
    img = img[y:h,x:w]

    gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    #gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    result = reader.readtext(gray)
    text = ""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
            text = res[1]
    #     text += res[1] + " "
    
    return str(text)



class DetectionPredictor(BasePredictor):
    video_name = None
    temp= None
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        predicted=""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        
        crop_characters=functions.detect_and_segment_plates_all_operations(orig_img,wpod_net)
        
        self.temp=functions.predict_with_CNN_model_plate(crop_characters,model_CNN,labels_CNN)
        if self.temp is not None and self.temp != "" and len(self.temp)>4:
            predicted=re.sub("[^a-zA-Z0-9]", "", self.temp)
        if predicted not in predicted_plate_CNN and predicted!="" and predicted!= " ":
            predicted_plate_CNN.append(predicted)
        
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch

        # video_name = self.video_name if self.video_name else self.data_path.stem
        # output_file = os.path.join('video_output', f'{video_name}_output.txt')
        # if os.path.exists(output_file):
        #     os.remove(output_file)
        

        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        for *xyxy, conf, cls in reversed(det):

            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                text_ocr = ocr_image(im0,xyxy)
                clean_plate = re.sub("[^a-zA-Z0-9]", "", text_ocr)
                

                similarity_threshold = 0.8  # Minimum benzerlik oranı

# Önceden eklenen clean_plate değerlerini tutmak için bir set veya liste oluşturun
                

                # Dosyaya ekleme yapmadan önce kontrol yapın
                if clean_plate is not None and len(clean_plate) != 0 and self.temp is not None and len(self.temp) != 0:
                    if clean_plate not in eklenen_plates:
                        similarity_ratio = difflib.SequenceMatcher(None, clean_plate, self.temp).ratio()

                        # if clean_plate==self.predicted_plate_CNN:
                        if similarity_ratio>similarity_threshold:
                            eklenen_plates.add(clean_plate)
                            #predicted_plate_CNN.append(self.temp)

                label = text_ocr
                predicted2=""
                
                if label == self.temp or functions.is_similar_plate(label,self.temp,0.6):
                    if self.temp is not None and self.temp != "" and len(self.temp)>4:
                        predicted2=re.sub("[^a-zA-Z0-9]", "", self.temp)
                    if predicted2 not in yolo_and_CNN and predicted2!="" and predicted2!= " ":
                        yolo_and_CNN.append(predicted2)
                        

                self.annotator.box_label(xyxy, label, color=colors(c, True))
                with open(os.path.join(str(self.save_dir), 'output.txt'), 'a') as f:
                    f.write(label + '\n')
            
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)
            
            

        return log_string
        
    

            

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    
    cfg.model = cfg.model or "best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
    with open('output-1.txt', 'a') as f:
        f.write('\n'.join(eklenen_plates) + '\n')
    print("Yolo before correction: ",eklenen_plates)
    print("CNN before correction: ",predicted_plate_CNN)
    arr1=[]
    arr1=predicted_plate_CNN.copy()
    arr2=functions.correct__final_predictions(arr1)
    print("Cnn after correction: ",arr2)
    print("CNN and Yolo before correction: ",yolo_and_CNN)
    print("CNN and Yolo after correction: ",functions.correct__final_predictions(yolo_and_CNN))
    
    
    
    
    end_time = time.time()
    execution_time = end_time - start_time

    print("Script execution time:", execution_time, "seconds")