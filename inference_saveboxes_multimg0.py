import os
import torch
import time
import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from lingua import Language, LanguageDetectorBuilder


def resize_with_aspect_ratio(image, target_width):
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True
    )
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
prompt = "[INST] <image>\nIn which language is this word used exactly as written, including accent marks? Answer with a single word. [/INST]"

input_dir = 'ppocr_img/ppocr_img/imgs'  # Input directory containing images
output_dir = 'cropped_images_llava'  # Output directory to save cropped images
os.makedirs(output_dir, exist_ok=True)

# Iterate through each image in the input directory
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    if os.path.isfile(img_path):
        start_time = time.time()
        # continue
        # Load the image
        image = cv2.imread(img_path)

        # Run OCR
        result = ocr.ocr(img_path, cls=True)

        # Iterate through each detected text box
        for idx in range(len(result)):
            res = result[idx]
            if res is None:
                continue
            for line_idx, line in enumerate(res):
                # Get the bounding box coordinates
                box = line[0]
                xmin = int(min([point[0] for point in box]))
                ymin = int(min([point[1] for point in box]))
                xmax = int(max([point[0] for point in box]))
                ymax = int(max([point[1] for point in box]))

                cropped_image = image[ymin:ymax, xmin:xmax]
                
                # Resize the cropped image
                resized_text_region = resize_with_aspect_ratio(cropped_image, target_width=400)
                inputs = processor(text=prompt, images=resized_text_region, return_tensors="pt")
                output = model.generate(**inputs, max_new_tokens=5)
                lang = processor.decode(output[0], skip_special_tokens=True)[123:]
                print("Predicted Language:", lang)

                # Draw BLIP answer on top of the bounding box image
                (text_width, text_height), baseline = cv2.getTextSize(lang, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                # Ensure the text position is within the image
                x, y = 10, 30  # Initial position
                image_height, image_width = resized_text_region.shape[:2]

                # Adjust the position if the text exceeds the image dimensions
                if x + text_width > image_width:
                    x = image_width - text_width - 10  # Adjust x position
                if y - text_height < 0:
                    y = text_height + 10  # Adjust y position

                # Draw the text on the image
                answer_img = cv2.putText(resized_text_region, lang, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Save the bounding box image with drawn answer
                cropped_img_name = f'{os.path.splitext(img_name)[0]}_cropped_{idx}_{line_idx}.jpg'
                bbox_img_save_path = os.path.join(output_dir, cropped_img_name)
                cv2.imwrite(bbox_img_save_path, answer_img)
                
                print(f"Time: {time.time() - start_time:.2f} s / img")

# Optionally, draw the OCR results on the original image and save it
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    if os.path.isfile(img_path):
        image = cv2.imread(img_path)
        result = ocr.ocr(img_path, cls=True)
        result = result[0]
        if result is None:
              continue
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        # languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.Japanese, Language.Russian, Language.Romanian, Language.Bulgarian, Language.Korean]
        detector =LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        txt_list = []
        for txt in txts:
                language = detector.detect_language_of(txt)
                txt_list.append(str(language))
                print(language)
        im_show = draw_ocr(image, boxes, txt_list, scores, font_path='ppocr_img/ppocr_img/fonts/simfang.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save(os.path.join('output_dir_paddle', f'{os.path.splitext(img_name)[0]}_result.jpg'))

