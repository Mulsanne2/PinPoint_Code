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
Image.MAX_IMAGE_PIXELS = None

### Dataset Information ###
data_path = "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/spdoc/"
qa_path = "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/dataset_final/spdoc/pinpoint_spdoc_val.json"

### Model Setting ###
pretrained = "/root/Desktop/workspace/kwon/pinpoint/LLaVA-NeXT/ckpt/spdoc/spdoc_pinpoint03" 
model_name = "llava_v1"
conv_template = "vicuna_v1"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="flash_attention_2") # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

with open(qa_path, "r", encoding="utf-8") as file:
    qa_data = json.load(file)

total_ANLS = 0
total_processed = 0
total_bbox_acc = 0
total_token_bf = 0
total_token_af = 0
total_image_ratio = 0.0
total_flops = 0.0
pbar = tqdm(qa_data)

for idx, entry in enumerate(pbar):
    image_path = data_path + entry['image']
    image = Image.open(image_path).convert("RGB")
    image_org_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    ques = entry['question']
    encompass_bbox = entry['encompass_bbox']
    answer_bbox = entry['answer_bbox']

    # question = DEFAULT_IMAGE_TOKEN + f"\n{ques}"
    # question = DEFAULT_IMAGE_TOKEN + f"\n{ques} \n Give me just an answer." #0.2625
    question = DEFAULT_IMAGE_TOKEN + f"\n{ques} \nAnswer the question using a single word or short phrase." #0.2777
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # print(prompt_question)
    input_ids, ques_tokens = tokenizer_image_token(prompt_question, ques, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)
    ques_tokens = ques_tokens.unsqueeze(0).to(device)
    image_sizes = [image.size]
    # with torch.no_grad():
    #     with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    #         with_flops=True,
    #         profile_memory=False,
    #         record_shapes=False
    #     ) as prof:
    cont, bbox_acc, tokens_bf, tokens_af, image_ratio = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=20,
        ques_tokens=ques_tokens,
        image_orgs=[image],
        image_org_sizes=[image_org_size],
        answer_bboxes=[answer_bbox],
        encompass_bboxes=[encompass_bbox],
        image_processor=image_processor,
    )
        # current_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    current_flops = 0

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    ANLS_Score = anls_score(prediction=text_outputs, gold_labels=entry['answers'])
    print(entry['question'])
    print(text_outputs)
    print(ANLS_Score)
    print("bbox_acc: ", bbox_acc)
    print("\n")

    # Update counters
    total_processed += 1
    total_ANLS += ANLS_Score
    total_bbox_acc += bbox_acc
    total_token_bf += tokens_bf
    total_token_af += tokens_af
    total_image_ratio += image_ratio
    total_flops += current_flops

    # Calculate and update the accuracy in the progress bar description
    if total_processed > 0:
        pbar.set_description(f"Processing | ANLS: {total_ANLS / total_processed:.3f} | Bbox Acc: {total_bbox_acc / total_processed:.2f} | Token ratio: {total_token_af / total_token_bf:.2f} | Image ratio: {total_image_ratio / total_processed:.2f} | FLOPs: {(total_flops / (total_processed *1e12)):.2f} TFLOPs")

print(f"\nFinal ANLS: {(total_ANLS / len(qa_data)):.4f}")
print(f"Final Bbox Acc: {(total_bbox_acc / len(qa_data)):.4f}")
print(f"Final Average Token BF: {(total_token_bf / len(qa_data)):.4f}")
print(f"Final Average Token AF: {(total_token_af / len(qa_data)):.4f}")
print(f"Final Token Ratio: {(total_token_af / total_token_bf):.4f}")
print(f"Final Image Ratio: {(total_image_ratio / len(qa_data)):.4f}")
print(f"Total FLOPs: {total_flops / (total_processed * 1e12):.2f} TFLOPs")