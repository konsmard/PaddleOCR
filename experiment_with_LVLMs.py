import os
import torch
import time
import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import inference
from transformers import CLIPProcessor, CLIPModel

torch.manual_seed(1234)


def resize_with_aspect_ratio(image, target_width):
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# prompt = "Question: In which language is this word used exactly as written, including accent marks? Answer with a single word. Answer:"

phrases = [
    "a photo of a word in English",
    "a photo of a word in Spanish",
    "a photo of a word in French",
    "a photo of a word in German",
    "a photo of a word in Italian",
    "a photo of a word in Portuguese",
    "a photo of a word in Romanian",
    "a photo of a word in Greek",
    "a photo of a word in Russian",
    "a photo of a word in Chinese",
    "a photo of a word in Japanese",
    "a photo of a word in Korean",
    "a photo of a word in Arabic",
    "a photo of a word in Hindi",
    "a photo of a word in Turkish",
    "a photo of a word in Persian",
    "a photo of a word in Dutch",
    "a photo of a word in Polish",
    "a photo of a word in Ukrainian",
    "a photo of a word in Czech",
    "a photo of a word in Swedish",
    "a photo of a word in Hungarian",
    "a photo of a word in Finnish",
    "a photo of a word in Norwegian",
    "a photo of a word in Danish"
]


phrases_dict = {i: phrase for i, phrase in enumerate(phrases)}
input_dir = 'ppocr_img/ppocr_img/imgs'  # Input directory containing images
output_dir = 'cropped_images3'  # Output directory to save cropped images
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    if os.path.isfile(img_path):
        start_time = time.time()
        
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
                
                inputs = processor(text=phrases, images=resized_text_region, return_tensors="pt", padding=True)

                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                lang_id = probs.argmax(1)
                # import ipdb; ipdb.set_trace()
                # result = pg.predict(resized_text_region,prompt)
                # results = model.infer(resized_text_region,prompt)[0]
                # lang = result
                # inputs = processor(text=prompt, images=resized_text_region, return_tensors="pt")
                # output = model.generate(**inputs, max_new_tokens=5)
                lang = phrases_dict[lang_id.item()][21:]
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
