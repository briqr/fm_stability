# classify_gender_only.py

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

import re 
import torch
from PIL import Image
import argparse
import os
import json # To save lists of indices

from tqdm import tqdm
from train import get_dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image


def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = args.dataset_name
 
    split = 'train'
    dataset = get_dataset(split=split, is_training=False, name=dataset_name)
    print(f'Length of dataset: {len(dataset)}')

    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model_name = "google/paligemma-3b-mix-224" 
    processor = AutoProcessor.from_pretrained(model_name)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        device_map="auto" 
    )

    output_dir = "gender_indices"
    os.makedirs(output_dir, exist_ok=True)
    
    male_indices = []
    female_indices = []
    unknown_indices = []
    
    temp_classification_data = [] 


    for idx, batch in enumerate(tqdm(loader, desc="Classifying gender")):
        
        real_im = batch['image'].to(device)
        
        real_im_processed = (((real_im * 0.5) + 0.5) * 255).type(torch.uint8)
        image_pil = to_pil_image(real_im_processed[0].cpu())

        # Only request gender attribute
        gender_prompt = "does the person in this image have male or female features? Please answer with 'Male' or 'Female'. If it's impossible to tell, answer 'Unknown'."
        
        full_prompt = f"<image><bos>{gender_prompt}"
        inputs = processor(text=full_prompt, images=image_pil, return_tensors="pt").to(model.device, torch.float16)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        num_input_tokens = inputs["input_ids"].shape[1]
        generated_tokens_only = generated_ids[0][num_input_tokens:]
        
        response_text = processor.decode(generated_tokens_only, skip_special_tokens=True).strip()
        
        cleaned_value = response_text.replace(gender_prompt, "").strip() 
        cleaned_value = re.sub(r"^(The\s+person\s+is\s+|The\s+person's\s+|It\s+is\s+|I\s+would\s+say\s+)(an?\s+)?", "", cleaned_value, flags=re.IGNORECASE).strip()
        cleaned_value = re.sub(r"(\.|\?|!|\,|\s*)$", "", cleaned_value).strip()
        cleaned_value = cleaned_value.lower()

        predicted_gender = "Cannot determine"
        if "unknown" in cleaned_value or "cannot determine" in cleaned_value or "not visible" in cleaned_value or not cleaned_value:
            predicted_gender = "Cannot determine"
        else:
            cleaned_words = cleaned_value.split()
            female_keywords = ["female", "woman", "girl", "ladies"]
            male_keywords = ["male", "man", "boy", "gentleman"]
            
            if any(word in cleaned_words for word in female_keywords):
                predicted_gender = "Female"
            elif any(word in cleaned_words for word in male_keywords):
                predicted_gender = "Male"
            
        if predicted_gender == "Male":
            male_indices.append(idx)
        elif predicted_gender == "Female":
            female_indices.append(idx)
        else:
            unknown_indices.append(idx)
        
        temp_classification_data.append({'index': idx, 'gender': predicted_gender})

    # Save indices to JSON files
    with open(os.path.join(output_dir, "train_male_indices.json"), 'w') as f:
        json.dump(male_indices, f)
    with open(os.path.join(output_dir, "train_female_indices.json"), 'w') as f:
        json.dump(female_indices, f)
    with open(os.path.join(output_dir, "train_unknown_indices.json"), 'w') as f:
        json.dump(unknown_indices, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='celebhq')
    args = parser.parse_args()
    main(args)
