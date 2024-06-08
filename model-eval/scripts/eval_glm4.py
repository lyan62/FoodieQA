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
import argparse

def build_input(mivqa, idx, data_dir, prompt=0, add_prompt_general=False):
    messages = []
    question = mivqa[idx]
    
    if prompt == 0:
        q = question["question"]
        q = q.replace("以下", "以上")
        for i in range(4):
            image = Image.open(os.path.join(data_dir, question["images"][i])).convert('RGB')
            messages.append({"role": "user", "image": image, "content":""})
        messages.append({"role": "user", "content": "根据以上四张图回答问题，他们分别为图A, 图B, 图C, 图D, 请从给定选项ABCD中选择一个最合适的答案。问题：{}, 答案为：图".format(q)})
    
    if prompt == 1:
        q = question["question"]
        q = q.replace("以下", "以上")
        for i in range(4):
            image = Image.open(os.path.join(data_dir, question["images"][i])).convert('RGB')
            if i == 3:
                messages.append({"role": "user", "image": image, "content": "图{}\n".format(chr(65+i)) +"根据以上四张图回答问题，请从给定选项ABCD中选择一个最合适的答案。问题：{}, 答案为：图".format(q)})
            else:
                messages.append({"role": "user", "image": image, "content": "图{}".format(chr(65+i))})
    if prompt == 2:
        q = question["question"]
        q = q.replace("以下", "以上")
        messages.append({"role": "user", "content": "根据以下四张图回答问题，请从给定选项ABCD中选择一个最合适的答案。"})
        for i in range(4):
            image = Image.open(os.path.join(data_dir, question["images"][i])).convert('RGB')
            if i ==3:
                messages.append({"role": "user", "image": image, "content": "图{}\n".format(chr(65+i)) + "问题：{}, 答案为：图".format(q)})
            else:
                messages.append({"role": "user", "image": image, "content": "图{}".format(chr(65+i))})
    if prompt == 3:
        q = question["question"]
        messages.append({"role": "user", "content": "Human: 问题{}，选项有: ".format(q)})
        for i in range(4):
            image = Image.open(os.path.join(data_dir, question["images"][i])).convert('RGB')
            if i==3:
                messages.append({"role": "assistant", "image": image, "content": "图{}\n".format(chr(65+i)) + "如果从给定选项ABCD中选择一个最合适的答案， 答案为：图"})
            else:
                messages.append({"role": "user", "image": image, "content": "图{}".format(chr(65+i))})
    return messages
        
        

def eval_question(mivqa, idx, prompt, data_dir, add_prompt_general=False):
    messages = build_input(mivqa, idx, data_dir, prompt=prompt)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors='pt')
    outputs = model.generate(inputs.to('cuda'), max_new_tokens=500)
    generated_texts = processor.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return {
        "response": generated_texts[0],
        "qid": mivqa[idx]["qid"]
    }
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", default="/scratch/project/dd-23-107/wenyan/cache")
    argparser.add_argument("--data_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie")
    argparser.add_argument("--eval_file", default="mivqa_filtered.json")
    argparser.add_argument("--prompt", type=int, default=3)
    argparser.add_argument("--out_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie/results/mivqa_res")
    argparser.add_argument("--model_name", default="THUDM/glm-4v-9b")
    args = argparser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    # load_model
    torch.cuda.empty_cache()

    model_name = "HuggingFaceM4/idefics2-8b"
    processor = AutoProcessor.from_pretrained(model_name, 
                                              cache_dir=os.environ["HF_HOME"], 
                                              do_image_splitting=False
                                              )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, cache_dir=os.environ["HF_HOME"], 
        device_map="auto", torch_dtype=torch.float16
        )

    # read_data
    data_dir = args.data_dir
    mivqa_file = args.eval_file
    prompt = args.prompt
    out_dir = args.out_dir
    
    mivqa = utils.read_mivqa(data_dir, mivqa_file)
    
    out_file_name = "mivqa_" + model_name.split("/")[-1] + "_prompt" + str(prompt) + ".jsonl"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Evaluating model on {} questions".format(len(mivqa)))
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for i in tqdm(range(len(mivqa))):
            res = eval_question(mivqa, i, data_dir=data_dir, prompt=prompt, add_prompt_general=True)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
            
    print("Saved model response to %s"%out_file_name)
    # accuracy = utils.get_accuracy(os.path.join(out_dir, out_file_name), 
    #                               mivqa, parse_fn=utils.parse_idefics)