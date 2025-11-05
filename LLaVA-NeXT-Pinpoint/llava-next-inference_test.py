from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
from tqdm import tqdm
import json
from anls import anls_score

### Dataset Information ###
data_path = "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/infographic/images/"
qa_path = "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/dataset_final/info/pinpoint_info_val.json"

### Model Setting ###
pretrained = "liuhaotian/llava-v1.6-vicuna-7b" #Vanila Setting
# pretrained = "/root/Desktop/workspace/kwon/pinpoint/LLaVA-NeXT/ckpt/llava-1.6-vicuna7b-test3" ### Full-Finetuning Setting
# adapter_path = "/root/Desktop/workspace/kwon/pinpoint/LLaVA-NeXT/ckpt/llava-1.6-vicuna7b-lora_e10" ### LoRA Setting
model_name = "llava_v1"
conv_template = "vicuna_v1"
device = "cuda"
device_map = "auto"

# tokenizer, model, image_processor, max_length = load_pretrained_model(adapter_path, pretrained, model_name, device_map=device_map, attn_implementation="flash_attention_2") # Add any other thing you want to pass in llava_model_args
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="flash_attention_2") # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

with open(qa_path, "r", encoding="utf-8") as file:
    qa_data = json.load(file)

total_ANLS = 0
total_processed = 0

pbar = tqdm(qa_data)

for entry in pbar:
    image_path = data_path + entry['image']
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    ques = entry['question']

    # question = DEFAULT_IMAGE_TOKEN + f"\n{ques}"
    # question = DEFAULT_IMAGE_TOKEN + f"\n{ques} \n Give me just an answer." #0.2625
    question = DEFAULT_IMAGE_TOKEN + f"\n{ques} \nAnswer the question using a single word or short phrase." #0.2777
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    print(prompt_question)
    input_ids = tokenizer_image_token(prompt_question, ques, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    ANLS_Score = anls_score(prediction=text_outputs, gold_labels=entry['answers'])
    print(entry['question'])
    print(text_outputs)
    print(ANLS_Score)
    print("\n")

    # Update counters
    total_processed += 1
    total_ANLS += ANLS_Score

    # Calculate and update the accuracy in the progress bar description
    if total_processed > 0:
        pbar.set_description(f"Processing | ANLS: {total_ANLS / total_processed:.2f}")

print(f"\nFinal ANLS: {(total_ANLS / len(qa_data)):.4f}")