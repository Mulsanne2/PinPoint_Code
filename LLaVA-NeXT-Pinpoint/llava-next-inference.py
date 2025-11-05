from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

# pretrained = "lmms-lab/llama3-llava-next-8b"
# model_name = "llava_llama3"
pretrained = "liuhaotian/llava-v1.6-vicuna-7b"
model_name = "llava_v1"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image_path = "/root/Desktop/workspace/kwon/infographic/vlm/llava/dataset/infographic/images/37313.jpeg"
image = Image.open(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
conv_template = "vicuna_v1"
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
# question = DEFAULT_IMAGE_TOKEN + "\nWhich social platform has heavy female audience? \n Just give me an answer."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=256,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])