from posixpath import splitext
from keras.models import model_from_json

import cv2 as cv2
from sklearn.preprocessing import LabelEncoder
from local_utils import detect_lp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pytesseract
import functions
from moviepy.editor import VideoFileClip
from collections import Counter
import difflib
import re
import calculation_functions
import visualize_functions
import shutil
from plakatanimasistemi.alg1_plaka_tespiti import plaka_konum_don
from plakatanimasistemi.alg2_plaka_tanima import plakaTani
import glob
global_i=0

import subprocess

def get_plates_from_photographs(pictures_file_path,wpod_net,model_type,model,label,visual_Type):
    predicted_plates=[]
    actual_plates=[]
    for filename in os.listdir(pictures_file_path):
        
        tempPath=os.path.join(pictures_file_path,filename)
        correct_filename,_=os.path.splitext(filename)
        actual_plates.append(correct_filename)

        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            crop_characters,plate_image,vehicle,LpImg,gray, blur, binary,thre_mor=functions.detect_and_segment_plates_all_operations(tempPath,wpod_net)
            
            if visual_Type:
                visualize_functions.draw_vehicle(vehicle,LpImg)
                visualize_functions.draw_morphed_plates(plate_image, gray, blur, binary,thre_mor)
                visualize_functions.draw_segmented_chars(crop_characters)
            
            if len(crop_characters) != 0:
                if model_type=="cnn":
                    predicted_plates.append(predict_with_CNN_model_plate(crop_characters,model,label))
                if model_type=="tess":
                    predicted_plates.append(predict_with_tesseract(plate_image))
                if model_type=="rcnn":
                    predicted_plates.append(predict_with_r_CNN_model_plate(crop_characters,model,label))
                
    return predicted_plates,actual_plates



def get_plates_from_video_with_frame(video_path,wpod_net,model_CNN,labels_CNN):
    output=extract_frames_from_video(video_path)
    temp_cnn=[]
    predicted=""
    
    for filename in os.listdir(output):
        tempPath=os.path.join(output,filename)

        crop_characters=detect_and_segment_plates_all_operations(tempPath,wpod_net)
        if len(crop_characters)!=0:
            
            temp=predict_with_CNN_model_plate(crop_characters,model_CNN,labels_CNN)
            if temp is not None and temp != "" and len(temp)>4:
                predicted=re.sub("[^a-zA-Z0-9]", "", temp)
            
            if predicted not in temp_cnn and predicted!="" and predicted!= " ":
                temp_cnn.append(predicted)
        # else:
        #     print(f"No characters found for the image .")
            
    for dosya in os.listdir(output):
        if dosya.endswith(".jpg") or dosya.endswith(".png") or dosya.endswith(".jpeg"):
            resim_yolu = os.path.join(output, dosya)
            os.remove(resim_yolu)
    
    return temp_cnn

def correct__final_predictions(predictions):
    for i in range(len(predictions)):
        if not predictions[i] or not any(char.isalpha() for char in predictions[i]):
            predictions[i] = None
        if predictions[i] is not None and not predictions[i][0].isdigit():
            predictions[i]=None



        for j in range(len(predictions)):
            if i != j and predictions[i] is not None and predictions[j] is not None and is_similar_plate(predictions[i], predictions[j], 0.6) :
                if len(predictions[i])>len(predictions[j]):
                    predictions[j] = None
                else:
                    predictions[i] = None

    predictions = [plate for plate in predictions if plate is not None]
    return predictions

def detect_and_segment_plates_all_operations(orig_img,wpod_net):
    vehicle, LpImg,cor =functions.get_plate(orig_img,wpod_net)
    plate_image, gray, blur, binary,thre_mor=functions.morph_pictures(LpImg)
    crop_characters_new =functions.segment_characters(plate_image)
    crop_characters_old=functions.segment_with_old_version(plate_image,binary,thre_mor)
    crop_characters = []
    
    if crop_characters_new is not None and crop_characters_old is not None:
        if len(crop_characters_old)>len(crop_characters_new) and len(crop_characters_old)<=8:
            crop_characters=crop_characters_old
        else:
            crop_characters=crop_characters_new
    
    return crop_characters,plate_image,vehicle,LpImg,gray, blur, binary,thre_mor

def extract_frames_from_video(video_path):
    video = cv2.VideoCapture(video_path)
    output="den"
    # Kare sayacı
    frame_count = 0
    i = 0

    # Videoyu kare kare işleyin
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Her 30 karede bir fotoğrafı kaydet
        if frame_count % 10 == 0:
            # Fotoğraf dosyasının adı ve yolunu oluşturun
            image_path = f"{output}/frame_{frame_count // 30}_{i}.jpg"

            # Fotoğrafı kaydet
            cv2.imwrite(image_path, frame)

        frame_count += 1
        i += 1

    # Videoyu serbest bırak
    video.release()
    return output

def segment_with_old_version(plate_image,binary,thre_mor):
        # creat a copy version "test_roi" of plat_image to draw bounding box
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # creat a copy version "test_roi" of plat_image to draw bounding box
    if plate_image is not None:
        test_roi = plate_image.copy()

        # Initialize a list which will be used to append charater image
    crop_characters = []

        # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.1: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    #print("Detect {} letters...".format(len(crop_characters)))
    # fig = plt.figure(figsize=(10,6))
    # plt.axis(False)
    # plt.imshow(test_roi)

    return crop_characters
def yeter():

    proje_dizini = r"D:\Projects\NewProjects\Plate_detect_and_recognize-master\plakatanimasistemi"
    dosya_adı = "pts.py"  # Çalıştırılacak dosyanın adını belirtin

    # Dosyayı çalıştırma
    process = subprocess.Popen(["python", dosya_adı], cwd=proje_dizini, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # Çıktıları kontrol etme
    if output:
        print(output.decode())
    if error:
        print(error.decode())


# def predict_with_random_forest(crop_chars):
    
    
#     # print("resim:",f"{input_file_path}/"+data)
#     img = cv2.imread(temp)
#     img = cv2.resize(img,(500,500))
#     plaka = plaka_konum_don(img)
#     plakaImg,plakaKarakter = plakaTani(img,plaka)
#     if plakaKarakter:
#         print("resimdeki plaka:",plakaKarakter)        
#         #plt.imshow(plakaImg)
#         # plt.show()
#     return plakaKarakter


def predict_randomforest_from_command():
    temp=r"D:\Projects\NewProjects\Plate_detect_and_recognize-master\plakatanimasistemi"
    os.chdir(temp)
    os.system(r"python D:\Projects\NewProjects\Plate_detect_and_recognize-master\plakatanimasistemi\PTS.py")



def plaka_temizle(txt_dosya_yolu):
    with open(txt_dosya_yolu, "r") as dosya:
        plakalar = dosya.readlines()

    cleaned_plates = []
    for plaka in plakalar:
        temiz_plaka = re.sub("[^a-zA-Z0-9]", "", plaka)
        cleaned_plates.append(temiz_plaka)

    plate_counts = Counter(cleaned_plates)
    
    #most_common_plates = plate_counts.most_common()
    #max_count = most_common_plates[0][1]  # En çok tekrar eden plakanın tekrar sayısı
    #unique_plates = [plate[0] for plate in most_common_plates if plate[1] == max_count]
    filtered_plates = [plate for plate in cleaned_plates if plate_counts[plate] > 2]
    unique_plates = list(set(filtered_plates))

    # Düzenlenmiş çıktıyı oluştur
    formatted_output = "\n".join(unique_plates)

# Düzenlenmiş çıktıyı dosyaya kaydet
    with open("formatted_output.txt", "w") as file:
        file.write(formatted_output)

    return formatted_output
    
def run_Yolo_and_CNN(file_name):
    project_folder = r'D:\Projects\NewProjects\Plate_detect_and_recognize-master\Automatic_Number_Plate_Detection_Recognition_YOLOv8\ultralytics\yolo\v8\detect'
    os.chdir(project_folder)
    os.system(f"python predict.py source='{file_name}'")
    

    output_folder = r'D:\Bitirme Projeleri\nreyolo\Automatic_Number_Plate_Detection_Recognition_YOLOv8\runs\detect'
    result_folders = os.listdir(output_folder)
    result_folders.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)))  # Tarihe göre sırala
    result_folder = result_folders[-1]

    source_path = os.path.join(output_folder, result_folder, file_name)
    txt_path = os.path.join(output_folder, result_folder, 'output.txt')

    output=plaka_temizle(txt_path)
    print(output)
    return source_path,txt_path

def carry_file_folder_from_folder(input_folder,output_folder):
    # "true_plates" klasörünü oluştur (eğer yoksa)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# "bad_outputs" klasöründeki tüm dosyaları dolaş
    for filename in os.listdir(input_folder):
        src_path = os.path.join(input_folder, filename)
        dest_path = os.path.join(output_folder, filename)

        shutil.move(src_path, dest_path)
    
def show_video(video_path):
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
    
    clip = VideoFileClip(video_path)
    frames = clip.iter_frames()
    fps = clip.fps

    for frame in frames:

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Video", frame)

        if cv2.waitKey(int(fps)) & 0xFF == ord("q") or cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def predict_with_CNN_model_plate(crop_characters,model,labels):
    #fig = plt.figure(figsize=(15,3))
    #cols = len(crop_characters)
    #grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

    if(crop_characters is None):
        print('No crop chars found for prediction')
        return None

    final_string = ''
    for i,character in enumerate(crop_characters):
        #fig.add_subplot(grid[i])
        title = np.array2string(functions.predict_from_model(character,model,labels))
        #plt.title('{}'.format(title.strip("'[]"),fontsize=20))
        final_string+=title.strip("'[]")
        #plt.axis(False)
        #plt.imshow(character,cmap='gray')
    if final_string and (not final_string[0].isdigit() or final_string[0] == ""):
        final_string = final_string[1:]

    return final_string

def predict_with_r_CNN_model_plate(crop_characters,model,labels):
    #fig = plt.figure(figsize=(15,3))
    #cols = len(crop_characters)
    #grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
    final_string = ''
    for i,character in enumerate(crop_characters):
        #fig.add_subplot(grid[i])
        title = np.array2string(functions.predict_from_model(character,model,labels))
        #plt.title('{}'.format(title.strip("'[]"),fontsize=20))
        final_string+=title.strip("'[]")
        #plt.axis(False)
        #plt.imshow(character,cmap='gray')
    if not final_string[0].isdigit():
        final_string = final_string[1:]

    return final_string

def configure_file_path(filePath):
    dosya_listesi = os.listdir(filePath)

    i = 1
    for dosya in dosya_listesi:
        if dosya.endswith(".jpg") or dosya.endswith(".jpeg") or dosya.endswith(".png"):
            resim_yolu = os.path.join(filePath, dosya)
            yeni_dosya_ad = str(i) + ".jpeg"
            yeni_dosya_yolu = os.path.join(filePath, yeni_dosya_ad)
            eski_dosya_yolu = os.path.join(filePath, dosya)
            if eski_dosya_yolu != yeni_dosya_yolu:
                if os.path.exists(yeni_dosya_yolu):
                    # Eğer hedef dosya zaten varsa, farklı bir adla kaydet
                    yeni_dosya_ad = str(i) + "_new.jpeg"
                    yeni_dosya_yolu = os.path.join(filePath, yeni_dosya_ad)
                os.rename(eski_dosya_yolu, yeni_dosya_yolu)
            i += 1

def correct_plate_for_tesseract(plate_text):
    if len(plate_text) < 4:
        return plate_text

    if len(plate_text) > 8:
        plate_text = plate_text[1:]

    if not plate_text[0].isdigit() and len(plate_text)>=8:
        plate_text = "3" + plate_text[1:]
    if not plate_text[0].isdigit():
        plate_text = plate_text[1:]
    elif len(plate_text) > 1 and not plate_text[1].isdigit():
        plate_text = plate_text[:1] + plate_text[2:]

    if plate_text[0]=="3":
        plate_text="3"+"4"+plate_text[2:]
    return plate_text

def configure_file_name(plate_image, filePath,file):
    
    file_path = os.path.join(filePath, file)   

    plate_text = pytesseract.image_to_string(plate_image, 
                                      config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()
    
    plate_text=correct_plate_for_tesseract(plate_text)
    
    # if plate_text[0].isdigit() and plate_text[1].isdigit() and plate_text[2].isdigit():
    #     plate_text=plate_text[0]+plate_text[1]+plate_text[3:]

    if(plate_text==file):
        return plate_text
    else:
        new_filename =  plate_text + os.path.splitext(file)[1]

        if new_filename != file:
            if os.path.exists(os.path.join(filePath, new_filename)):
                print(f"File with name {new_filename} already exists.")
            else:
                os.rename(file_path, os.path.join(filePath, new_filename))
    
    return plate_text


def predict_with_tesseract(plate_image):
    plate_text = pytesseract.image_to_string(plate_image, 
                                      config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()
    plate_text=correct_plate_for_tesseract(plate_text)

    return plate_text

def calculate_accuracy_for_configure_input_pictures(actual_plates,predicted_plate,filename,pictures_file_path):
    global global_i
    accuracy,a,b,c,d,e=calculation_functions.calculate_metrics(actual_plates,predicted_plate)
    if(accuracy<0.3):
        print(f"Düşük acc oldugu için {filename} taşındı.")
        shutil.move(os.path.join(pictures_file_path, filename), os.path.join("bad_outputs", filename))
        global_i=global_i+1
        return False
    return True

def calculate_accuracy_for_configure_for_video_frame(actual_plates,predicted_plate,filename,pictures_file_path):
    global global_i
    accuracy,a,b,c,d,e=calculation_functions.calculate_metrics(actual_plates,predicted_plate)
    if(accuracy<0):
        print(f"Düşük acc oldugu için {filename} taşındı.")
        #shutil.move(os.path.join(pictures_file_path, filename), os.path.join("bad_outputs", filename))
        os.remove(os.path.join(pictures_file_path, filename))
        global_i=global_i+1
        return False
    return True

def calculate_accuracy_for_configure_input_pictures_for_tesseract(actual_plates,predicted_plate,filename,pictures_file_path):
    global global_i
    accuracy,a,b,c,d,e=calculation_functions.calculate_metrics(actual_plates,predicted_plate)
    if(accuracy<0.001):
        print(f"Düşük acc oldugu için {filename} taşındı.")
        shutil.move(os.path.join(pictures_file_path, filename), os.path.join("bad_outputs", filename))
        global_i=global_i+1
        return False
    return True

    
   


def morph_pictures(LpImg):
    if len(LpImg) == 0:
        return None, None, None, None, None
    
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    return plate_image,gray,blur,binary,thre_mor

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

def load_train_model(char,weight,classes):
    
    json_file = open(char, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(loaded_model_json)
    model.load_weights(weight)
    print("[INFO] Model loaded successfully...")
    
    labels = LabelEncoder()
    labels.classes_ = np.load(classes)
    print("[INFO] Labels loaded successfully...")

    return model,labels

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:],verbose=0))])
    return prediction

def preprocess_image(img,resize=False):
    
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        if resize:
            img = cv2.resize(img, (224,224))
        return img
    
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions

def is_similar_plate(plate1, plate2, similarity_threshold=0.8):
    similarity = difflib.SequenceMatcher(None, plate1, plate2).ratio()
    return similarity >= similarity_threshold

def get_plate(image_path,model, Dmax=608, Dmin = 600):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(model, vehicle, bound_dim, lp_threshold=0.5)
    
    return vehicle, LpImg, cor

# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting images
def segment_characters(image) :

    if image is None:
        print('No plate is found for segmentation')
        return None 
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list



def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    if cnts is None or len(cnts)!=0:
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts