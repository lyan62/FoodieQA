import requests
import torch
from PIL import Image
from io import BytesIO
import yaml
import os
import json
from tqdm import tqdm 

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

import utils

PROMPT_GENERAL = "请从给定选项ABCD中选择一个最合适的答案。"

# format inputs
def format_image_input(img_idx, template=0):
    idx2choice = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }
    if template == 0:
        img_input = {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                    ]
        }
    if template == 1:
        img_input = {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "图"+idx2choice[img_idx]},
                    ]
        }
    if template == 2:
        img_input = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "图"+idx2choice[img_idx]},
                        {"type": "image"},
                    ]
        }
    return img_input

def format_text_prompt(text_prompt):
    return  {
        "role": "user",
        "content": [{"type": "text", "text": text_prompt},]
        }
    

def format_text_input(question, template=0, add_prompt_general=False):
    q = question["question"]
    if template == 0:
        if "以下" in q:
            q=q.replace("以下", "以上")   
        
        if add_prompt_general:
            text_prompt = "根据以上四张图回答问题，他们分别为图A, 图B, 图C, 图D."\
                +PROMPT_GENERAL+"问题：{}, 答案为：图".format(q)
        else:
            text_prompt = "根据以上四张图回答问题，他们分别为图A, 图B, 图C, 图D, 问题：{}, 答案为：图".format(q)
        
        text_input = format_text_prompt(text_prompt)
    if template == 1:
        if "以下" in q:
            q=q.replace("以下", "以上")
        
        if add_prompt_general:
            text_prompt = "根据以上四张图回答问题,"+PROMPT_GENERAL+"问题：{}, 答案为：图".format(q)
        else:
            text_prompt = "根据以上四张图回答问题, 问题：{}, 答案为：图".format(q)
        text_input = format_text_prompt(text_prompt)
    if template == 2:
        if "以上" in q:
            q=q.replace("以上", "以下")
        if add_prompt_general:
            text_prompt = "根据以下四张图回答问题,"+PROMPT_GENERAL
        else:
            text_prompt = "根据以下四张图回答问题,"
        text_input = format_text_prompt(text_prompt)
    if template == 3:
        text_prompt = "问题{}，选项有: ".format(q)
        text_input = format_text_prompt(text_prompt)
    return text_input
        

def build_input(mivqa, idx, prompt=0):
    messages = []
    question = mivqa[idx]
    if prompt == 0 or prompt ==1:
        for i in range(4):
            img_input = format_image_input(i, template=prompt)
            messages.append(img_input)
        text_input = format_text_input(question, template=prompt)
        messages.append(text_input)
        
        images = [load_image(os.path.join(data_dir, img)) for img in question["images"]]
    if prompt ==2:
        text_input = format_text_input(question, template=2)
        messages.append(text_input)
        for i in range(4):
            img_input = format_image_input(i, template=1)
            messages.append(img_input)
        messages.append(format_text_prompt("问题：{}, 答案为：图".format(question["question"])))
    if prompt == 3:
        text_input = format_text_input(question, template=3)
        messages.append(text_input)
        for i in range(4):
            img_input = format_image_input(i, template=2)
            messages.append(img_input)
        messages.append(format_text_prompt("如果从给定选项ABCD中选择一个最合适的答案：{}, 答案为：图".format(question["question"])))
        
    return messages, images


def eval_question(mivqa, idx, prompt):
    messages, images = build_input(mivqa, idx, prompt=prompt)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to() for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return {
        "response": generated_texts,
        "qid": mivqa[idx]["qid"]
    }
    
if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    os.environ['HF_HOME'] = config["cache_dir"] #'/scratch3/wenyan/cache'


    # load_model
    torch.cuda.empty_cache()

    model_name = config.get("model_name", "HuggingFaceM4/idefics2-8b")
    processor = AutoProcessor.from_pretrained(model_name, 
                                              cache_dir=os.environ["HF_HOME"], 
                                              do_image_splitting=False
                                              )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, cache_dir=os.environ["HF_HOME"], 
        device_map="auto", torch_dtype=torch.float16
        )

    # read_data
    data_dir = config["data_dir"]
    mivqa_file = config["eval_file"]
    prompt = config["prompt"]
    out_dir = config["out_dir"]
    
    mivqa = utils.read_mivqa(data_dir, mivqa_file)
    
    out_file_name = "mivqa_" + model_name.split("/")[-1] + "_prompt" + str(prompt) + ".jsonl"
    
    print("Evaluating model on {} questions".format(len(mivqa)))
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for i in tqdm(range(len(mivqa))):
            res = eval_question(mivqa, i, img_template=0, text_template=0)
            f.write(json.dumps(res)+"\n")
            
    print("Calculate accuracy")
    with open(os.path.join(out_dir, out_file_name), "r") as f:
        accuracy = utils.get_accuracy(f, mivqa, parse_fn=utils.parse_idefics)