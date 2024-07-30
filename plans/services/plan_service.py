# myapp/views.py
#from openai import OpenAI
import numpy as np
import openai
from ultralytics import YOLO
import os
from django.conf import settings
import cv2
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
import datetime
import cv2 # type: ignore


from PIL import Image

# Update the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

openai.api_key = ""
MODEL = YOLO(os.path.join(settings.BASE_DIR, 'plans/zone_detectors/zone_detector_v1.pt'))
print(MODEL)  # Debug statement to check initialization


#PROMPT = "**Prompt for Scaling Factor**\n\nLocate the line in the image which has the dimensions 640x640 at the top or bottom where the real-life length of the pixelated line is given. Scan the image for that line, measure its pixelated length, compare it with the real-life length given, and find the scaling factor. The answer should be provided as a single number which is the scaling factor.\n\n**Calculation Process**\n\n1. **Locate the line with the real-life length mentioned:**\n   - In the provided architectural plan, the information about the real-life length of the total width of the building is provided at the top of the plan\n   - In the provided architectural plan, the information about the real-life length of the total length of the building is provided on the left of the plan\n   - In 2. I will provide the boxes predicted by a YOLO model\n   - use all of this information to calculate accurate pixelated length and scaling factor.\n\n2. **boxes predicted by a YOLO model on the image**\n\n   - {{BOXES}}\n\n2. **Measure the pixel length of that line:**\n   - Using an appropriate tool or software, measure the pixel length of both the segments, i.e the length and breadth\n   - Ensure the measurement tool is accurate and consistent.\n   - use both the segments to cross-reference the calculated pixelated length and ensure it's accuracy\n   - do not assume the pixelated length but **calculate it yourself**\n\n3. **Double-check the pixel length measurement:**\n   - Verify the pixel length measurement by measuring it multiple times to ensure consistency.\n   - Confirm that the measurements are precise and remeasure if necessary.\n\n4. **Calculate the scaling factor:**\n   \n   \[ \text{Scaling Factor} = \frac{\text{Real-life Length}}{\text{Pixel Length}} \]\n\n5. **Validate the result:**\n   - Cross-check the calculated scaling factor against another dimension in the image as a secondary validation.\n   - Ensure the scaling factor aligns with both measurements.\n\n6. **Result Formatting**\n   - Don't explain the process of how you came up with the scaling factor\n   - your response should look like this:\n   - {{SCALING FACTOR}} = .....\n"
#PROMPT = "**Prompt for Scaling Factor**\n\nLocate the line in the image at the top or bottom where the real-life length of the pixelated line is given. Scan the image for that line, measure its pixelated length, compare it with the real-life length given, and find the scaling factor. The answer should be provided as a single number which is the scaling factor.\n\n---\n\n**Calculation Process**\n\n1. **Locate the line with the real-life length mentioned:**\n   - In the provided architectural plan, the real-life length of the total width of the building is given as 34,000 mm (34 meters). This information is provided at the top of both the ground floor and first floor plans.\n\n2. **Measure the pixel length of that line:**\n   - Using an appropriate tool or software, measure the pixel length of the 34,000 mm segment. Assume the segment from the left edge to the right edge of the building measures approximately 840 pixels.\n\n3. **Calculate the scaling factor:**\n   \n   \[ \text{Scaling Factor} = \frac{\text{Real-life Length}}{\text{Pixel Length}} \]\n\n4. **Result Formatting**\n   - Don't explain the process of how you came up with the scaling factor\n   - your response should look like this:\n   - {{SCALING FACTOR}} = ....."

def calculate_areas(image_path):
    #image_url = "https://i.ibb.co/n89PKd2/36-Bed-Girsl-Hostel-NIFT-28-7-2021-images-3.jpg"

    bboxes, scores, cls_names = predict(image_path)
    scaling_factor = extract_scaling_factor(image_path)
    areas = []
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        conf = scores[idx]
        cls = cls_names[idx]
        #polygon = Polygon((x1, y1), (x2, y2))
        width = ((x2 - x1) * scaling_factor) / 1000
        height = ((y2 - y1) * scaling_factor) / 1000
        area = width * height
        areas.append({
            'class': cls,
            'bbox': [x1, y1, x2, y2],
            'area': area,
            'confidence': conf            
        })

    plot(image_path, areas)
    return areas

    print(areas)


def predict(image_path):
    print(f"Predicting for image: {image_path}")  # Debug statement
    results = MODEL(image_path)  # Run the model on the input image
    # Process the results
    predictions = results[0].boxes.xyxy.numpy()
    scores = results[0].boxes.conf.numpy()
    classes = results[0].boxes.cls.numpy()
    names = results[0].names
    cls_names = []
    for cls in classes:
        cls_names.append(names[cls])
    return predictions, scores, cls_names


def plot(image_path, new_boxes_subset):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    print(image_path)
    # file.copy(image_path, r"C:\\Users\\vinay\\Downloads\\output")

    for box in new_boxes_subset:
        x_min, y_min, x_max, y_max = box['bbox']
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, f"Area: {box['area']:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    filelocation= r"C:\\Users\\vinay\\Downloads\\output\\output.png"
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    extension='.png'
    base_path = r"C:/Users/vinay/Downloads/outputarchive"
    filename_prefix = 'processed_image'
    # Create the filename with the timestamp
    filename = f"{filename_prefix}_{timestamp}{extension}"
    file_path = f"{base_path}/{filename}"
    plt.savefig(filelocation)
    plt.savefig(file_path)

    #plt.show()


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Noise reduction
    image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    # Adaptive thresholding
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_image


def extract_text(image):
    custom_config = r'--oem 3 --psm 6'
    text_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
    return text_data


def extract_lines(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines


def merge_lines(lines):
    if lines is None:
        return []

    lines = lines.reshape(-1, 4)
    merged_lines = []
    for (x1, y1, x2, y2) in lines:
        added = False
        for i, (mx1, my1, mx2, my2) in enumerate(merged_lines):
            if np.abs(my1 - y2) < 50 or np.abs(my2 - y1) < 50:
                merged_lines[i] = (min(mx1, x1, mx2, x2), my1, max(mx1, x1, mx2, x2), my2)
                added = True
                break
        if not added:
            merged_lines.append((x1, y1, x2, y2))
    return np.array(merged_lines)


def find_largest_number(text_data):
    numbers = [(int(text.strip()), idx) for idx, text in enumerate(text_data['text']) if text.strip().isdigit()]
    if numbers:
        largest_number, index = max(numbers, key=lambda x: x[0])
        return largest_number, index
    else:
        return None, None


def find_reference_line(text_data, lines, target_idx):
    x, y, w, h = (text_data['left'][target_idx], text_data['top'][target_idx],
                  text_data['width'][target_idx], text_data['height'][target_idx])

    mid_x = x + w // 2
    mid_y = y + h // 2
    max_distance = 30

    lines = lines.reshape(-1, 4)  # Ensure lines are in the correct shape
    valid_lines = [line for line in lines if abs(line[1] - line[3]) < 50]  # Ensure horizontal lines
    merged_lines = merge_lines(np.array(valid_lines))

    closest_line = None
    min_distance = float('inf')

    for x1, y1, x2, y2 in merged_lines:
        line_mid_x = (x1 + x2) // 2
        line_mid_y = (y1 + y2) // 2
        distance = abs(line_mid_y - mid_y)
        if distance < max_distance and distance < min_distance:
            closest_line = (x1, y1, x2, y2)
            min_distance = distance

    return closest_line


def calculate_scaling_factor(largest_number, reference_line):
    if reference_line is None:
        return None

    x1, y1, x2, y2 = reference_line
    pixel_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    real_length = largest_number
    scaling_factor = real_length / pixel_length
    return scaling_factor


def visualize_lines(image_path, text_data, target_idx, merged_lines, name='debug_image.png'):
    image = cv2.imread(image_path)
    x, y, w, h = (text_data['left'][target_idx], text_data['top'][target_idx],
                  text_data['width'][target_idx], text_data['height'][target_idx])

    mid_x = x + w // 2
    mid_y = y + h // 2

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(image, (mid_x, mid_y), 5, (255, 0, 0), -1)

    for line in merged_lines:
        x1, y1, x2, y2 = line
        line_mid_x = (x1 + x2) // 2
        line_mid_y = (y1 + y2) // 2
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(image, (line_mid_x, line_mid_y), 5, (0, 255, 255), -1)

    cv2.imwrite(name, image)


def extract_scaling_factor(image_path):
    binary_image = preprocess_image(image_path)
    text_data = extract_text(binary_image)
    largest_number, largest_number_idx = find_largest_number(text_data)

    if largest_number is None:
        print("No numeric values found in the text data.")
        return

    lines = extract_lines(binary_image)
    valid_lines = [line[0] for line in lines if abs(line[0][1] - line[0][3]) < 50]  # Ensure horizontal lines
    merged_lines = merge_lines(np.array(valid_lines))

    visualize_lines(image_path, text_data, largest_number_idx, np.array(valid_lines), 'debug_image-non-merged.png')
    visualize_lines(image_path, text_data, largest_number_idx, merged_lines, 'debug_image-merged.png')

    reference_line = find_reference_line(text_data, merged_lines, largest_number_idx)
    scaling_factor_value = calculate_scaling_factor(largest_number, reference_line)

    if scaling_factor_value is not None:
        print(f"The calculated scaling factor is: {scaling_factor_value}, reference line is: {reference_line}")
    else:
        print("No suitable reference line found for the largest number. Please check the input image and try again.")

    return scaling_factor_value

# def calculate_scaling_factor(image_url, prompt):
#     # file_response = openai.files.create(
#     #     file=open(image_path, "rb"),
#     #     purpose="fine-tune"
#     # )
#     response = openai.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt,
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": image_url,
#                         },
#                     },
#                 ],
#             }
#         ],
#         max_tokens=800,
#     )
#
#     return float(response.choices[0].message.content.split("{{SCALING FACTOR}} = ")[1])

# def predict(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         image_file = request.FILES['image']
#         results = model(image_file)  # Run the model on the input image
#
#         # Process the results
#         predictions = results[0].boxes.xyxy.numpy()  # Example: get bounding boxes
#         scores = results[0].boxes.conf.numpy()       # Example: get scores
#
#         response_data = {
#             'predictions': predictions.tolist(),
#             'scores': scores.tolist()
#         }
#         return JsonResponse(response_data)
#
#     return JsonResponse({'error': 'Invalid request'}, status=400)
# image_path = "/Users/kash/Downloads/smallpdf-convert-20240616-164931/36 Bed Girsl Hostel NIFT 28-7-2021-images-3.jpg"


# Function to save an image with a timestamp
def save_image_with_timestamp(image, base_path, filename_prefix, extension='.png'):
    if image is None:
        print("Error: Image is None, ensure the image path is correct and the image is read successfully.")
        return
    
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create the filename with the timestamp
    filename = f"{filename_prefix}_{timestamp}{extension}"
    
    # Create the full file path
    file_path = f"{base_path}/{filename}"
    
    # Save the image
    cv2.imwrite(file_path, image)
    
    print(f"Image saved to {file_path}")
