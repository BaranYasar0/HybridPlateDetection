import shutil
import os

def calculate_true_positives(true_plates, predicted_plates):
    true_positives = 0
    
    for true_plate, predicted_plate in zip(true_plates, predicted_plates):
        for character in true_plate:
            if character in predicted_plate:
                true_positives += 1
    
    return true_positives

def calculate_false_positives(true_plates, predicted_plates):
    false_positives = 0
    
    for true_plate, predicted_plate in zip(true_plates, predicted_plates):
        true_characters = set(true_plate)
        predicted_characters = set(predicted_plate)
        false_positives += len(predicted_characters - true_characters)
    
    return false_positives

def calculate_false_negatives(true_plates, predicted_plates):
    false_negatives = 0
    
    for true_plate, predicted_plate in zip(true_plates, predicted_plates):
        true_characters = set(true_plate)
        predicted_characters = set(predicted_plate)
        false_negatives += len(true_characters - predicted_characters)
    
    return false_negatives

def calculate_total_characters(true_plates):
    total_characters = 0
    
    for true_plate in true_plates:
        total_characters += len((true_plate))
    if total_characters>100:
        total_characters=total_characters

    return total_characters

def calculate_metrics(true_plates, predicted_plates):
    true_positives = calculate_true_positives(true_plates, predicted_plates)
    false_positives = calculate_false_positives(true_plates, predicted_plates)
    false_negatives = calculate_false_negatives(true_plates, predicted_plates)
    total_characters = calculate_total_characters(true_plates)

    accuracy = (true_positives / total_characters) if total_characters > 0 else 0
    precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # print("False Positives:", false_positives)
    # print("False Negatives:", false_negatives)
    
    return accuracy, precision, recall, f1_score,true_positives,total_characters


def calculate_metrics_for_CNN(true_plates,predicted_plates):
    print("Metrics for CNN")
    acc,prec,rec,f1,true_positives,total_characters=calculate_metrics(true_plates,predicted_plates)
    print_metrics(acc,prec,rec,f1,true_positives,total_characters)
    return acc,prec,rec,f1,true_positives,total_characters

def calculate_metrics_for_R_CNN(true_plates,predicted_plates):
    print("Metrics for R-CNN")
    acc,prec,recall,f1,true_positives,total_characters=calculate_metrics(true_plates,predicted_plates)
    print_metrics(acc,prec,recall,f1,true_positives,total_characters)
    return acc,prec,recall,f1,true_positives,total_characters


def calculate_metrics_for_tesseract(true_plates,predicted_plates):
    print("Metrics for tesseract")
    acc,prec,recall,f1,true_positives,total_characters=calculate_metrics(true_plates,predicted_plates)
    print_metrics(acc,prec,recall,f1,true_positives,total_characters)
    return acc,prec,recall,f1,true_positives,total_characters


def print_metrics(acc,prec,recall,f1,true_positives,total_characters):
    print("True Positives:", true_positives)
    print(f"Total Characters: {total_characters} \n")
    print("Accuracy: %",acc*100)  
    print("Precision: %", prec*100)  
    print("Recall: %" , recall*100)  
    print(f"F1 Score: % {f1*100}\n")