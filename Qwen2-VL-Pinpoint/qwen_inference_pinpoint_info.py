from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import copy
import torch
from tqdm import tqdm
import json
from anls import anls_score
import torch.profiler
### Dataset Information ###
data_path = "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/infographic/images/"
qa_path = "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/dataset_final/info/pinpoint_info_val.json"
device_map = "auto"
model_path = "Qwen/Qwen2-VL-7B-Instruct"
# model_path2 = "/root/Desktop/workspace/kwon/pinpoint/qwen_ft/ckpt/qwen_info_01"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, trust_remote_code=True,torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map=device_map
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


with open(qa_path, "r", encoding="utf-8") as file:
    qa_data = json.load(file)

total_ANLS = 0
total_processed = 0
total_len = 0
total_time = 0.0 
total_flops = 0.0
total_bbox_acc = 0.0
total_token_bf = 0
total_token_af = 0
total_image_ratio = 0.0

pbar = tqdm(qa_data)
for entry in pbar:
    image_path = data_path + entry['image']
    image = Image.open(image_path).convert("RGB")
    ques = entry['question']

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": f"{ques} \n Give me just an answer."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    ques = processor.tokenizer([ques])
    inputs["ques"] = torch.tensor(ques['input_ids'])
    inputs["image_org"] = [image]
    inputs["image_org_size"] = [(image.width, image.height)]
    inputs["encompass_bbox"] = [entry["encompass_bbox"]]
    inputs["answer_bbox"] = [entry["answer_bbox"]]
    inputs = inputs.to("cuda")


    # Inference: Generation of the output

    # with torch.no_grad():
    #     with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        #     with_flops=True,
        #     profile_memory=False,
        #     record_shapes=False
        # ) as prof:
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # current_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    current_flops = 0 

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    text_outputs = output_text[0]

    ANLS_Score = anls_score(prediction=text_outputs, gold_labels=entry['answers'])
    print(entry['question'])
    print(text_outputs)
    print(ANLS_Score)
    print("\n")
    if model.model.bbox_acc is not None:
        bbox_acc = model.model.bbox_acc
        image_ratio = model.model.image_ratio
        tokens_bf = model.model.tokens_bf
        tokens_af = model.model.tokens_af
    else:
        bbox_acc = 0
        image_ratio = 0
        tokens_bf = 1
        tokens_af = 1
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
print(f"Total FLOPs: {total_flops / (total_processed * 1e12):.2f} TFLOPs")
print(f"Total Image ratio: {total_image_ratio / len(qa_data):.2f}")
