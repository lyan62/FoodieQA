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

PROMPT_GENERAL = "请从给定选项ABCD中选择一个最合适的答案。"


def build_input(mivqa, idx, template=0):
    question = mivqa[idx]
    images = [load_image(os.path.join(data_dir, img)) for img in question["images"]]
    q = question["question"]
    idx2choice = {0:"A", 1:"B", 2:"C", 3:"D"}
    
    if template == 0:
        q = q.replace("以下", "以上")
        query_list = [{"type": "image"} for image in question["images"]]
        query_list.append({"type": "text", "text": "根据以上四张图回答问题，他们分别为图A, 图B, 图C, 图D, " + PROMPT_GENERAL + "问题：{}, 答案为：图".format(q)})
    
    if template == 1:
        q = q.replace("以下", "以上")
        query_list = []
        images = question["images"]
        for i in range(len(images)):
            query_list = [{"type": "image"} for image in question["images"]]
            query_list.append({"type": "text", "text" : "图{}\n".format(idx2choice[i])})
        query_list.append({"text": "根据以上四张图回答问题," + PROMPT_GENERAL + "问题：{}, 答案为：图".format(q)})
        
    if template == 2:
        q = q.replace("以上", "以下")
        query_list = [{"type": "text", "text":"根据以下四张图回答问题," + PROMPT_GENERAL}]
        images = question["images"]
        
        for i in range(len(images)):
            query_list.append({"type": "text", "text" : "图{}".format(idx2choice[i])})
            query_list = [{"type": "image"} for image in question["images"]]
        query_list.append({"type": "text", "text": "问题：{}, 答案为：图".format(q)})
    
    if template == 3:
        q = q.replace("以下", "以上")
        query_list = [{"type": "text", "text": "Human: 问题{}，选项有: ".format(q)}]
        for i in range(len(images)):
            query_list.append({"type": "text", "text" : "图{}".format(idx2choice[i])})
            query_list.append({"image" : os.path.join(data_dir, images[i])})
        query_list.append({"type": "text", "text": "Assistant: 如果从给定选项ABCD中选择一个最合适的答案， 答案为：图"})
    
    messages = [
        {
            "role": "user",
            "content": query_list
        }    
    ]
    return messages, images

def eval_question(mivqa, idx, prompt, add_prompt_general=False):
    messages, images = build_input(mivqa, idx, prompt=prompt)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to() for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "response": generated_texts,
        "qid": mivqa[idx]["qid"]
    }
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", default="/scratch/project/dd-23-107/wenyan/cache")
    argparser.add_argument("--data_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie")
    argparser.add_argument("--eval_file", default="mivqa_filtered.json")
    argparser.add_argument("--prompt", type=int, default=0)
    argparser.add_argument("--out_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie/results")
    argparser.add_argument("--model_name", default="TIGER-Lab/Mantis-8B-Idefics2")
    args = argparser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    # load_model
    torch.cuda.empty_cache()

    model_name = args.model_name
    processor = AutoProcessor.from_pretrained(model_name, 
                                          cache_dir=os.environ["HF_HOME"]) # do_image_splitting is False by default
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, cache_dir=os.environ["HF_HOME"],
        device_map="auto"
    )
    
    # generation_kwargs = {
    #     "max_new_tokens": 1024,
    #     "num_beams": 1,
    #     "do_sample": False
    # }
    
    # read_data
    data_dir = args.data_dir
    mivqa_file = args.eval_file
    prompt = args.prompt
    out_dir = args.out_dir
    
    mivqa = utils.read_mivqa(data_dir, mivqa_file)
    
    out_file_name = "mivqa_" + 'mantis' + "_prompt" + str(prompt) + ".jsonl"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Evaluating model on {} questions".format(len(mivqa)))
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for i in tqdm(range(len(mivqa))):
            res = eval_question(mivqa, i, prompt=prompt, add_prompt_general=True)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
            
    print("Saved model response to %s, Calculate accuracy"%out_file_name)
    accuracy = utils.get_accuracy(os.path.join(out_dir, out_file_name), 
                                  mivqa, parse_fn=utils.parse_mantis)