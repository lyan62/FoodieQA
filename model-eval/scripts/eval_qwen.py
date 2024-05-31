import requests
import torch
from PIL import Image
from io import BytesIO
import yaml
import os
import json
from tqdm import tqdm 
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 

import utils

PROMPT_GENERAL = "请从给定选项ABCD中选择一个最合适的答案。"

def get_query_list(question, data_dir, template=0):
    q = question["question"].strip()
    if template == 0:
        q = q.replace("以下", "以上")
        query_list = [{"image": os.path.join(data_dir, image)} for image in question["images"]]
        query_list.append({"text": "根据以上四张图回答问题，他们分别为图A, 图B, 图C, 图D" + PROMPT_GENERAL + "问题：{}, 答案为：图".format(q)})
    if template == 1:
        q = q.replace("以下", "以上")
        query_list = []
        images = question["images"]
        idx2choice = {0:"A", 1:"B", 2:"C", 3:"D"}
        for i in range(len(images)):
            query_list.append({"image" : os.path.join(data_dir, images[i])})
            query_list.append({"text" : "图{}\n".format(idx2choice[i])})
        query_list.append({"text": "根据以上四张图回答问题," + PROMPT_GENERAL + "问题：{}, 答案为：图".format(q)})
    if template == 2:
        query_list = [{"text":"根据以下四张图回答问题," + PROMPT_GENERAL}]
        images = question["images"]
        idx2choice = {0:"A", 1:"B", 2:"C", 3:"D"}
        for i in range(len(images)):
            query_list.append({"text" : "图{}".format(idx2choice[i])})
            query_list.append({"image" : os.path.join(data_dir, images[i])})
        query_list.append({"text": "问题：{}， 答案为：图".format(question["question"])})
    if template == 3:
        q = q.replace("以下", "以上")
        query_list = [{"image": os.path.join(data_dir, image)} for image in question["images"]]
        query_list.append({"text": "根据以上四张图回答问题, 问题：{}, 答案为：Picture".format(question["question"])})
    return query_list

def eval_qwen(mivqa, i, template=0):
    question = mivqa[i]
    query_list = get_query_list(question, data_dir, template=template)
    query = tokenizer.from_list_format(query_list)
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
    return {
        "response": response,
        "qid": mivqa[i]["qid"]
    }
    
if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    os.environ['HF_HOME'] = config["cache_dir"] #'/scratch3/wenyan/cache'


    # load_model
    torch.cuda.empty_cache()
    # Downloading model checkpoint to a local dir model_dir
    model_dir = snapshot_download('qwen/Qwen-VL', cache_dir=os.environ['HF_HOME'])
    # model_dir = snapshot_download('qwen/Qwen-VL-Chat')


    # Loading local checkpoints
    # trust_remote_code is still set as True since we still load codes from local dir instead of transformers
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, do_image_splitting=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    # read_data
    data_dir = config["data_dir"]
    mivqa_file = config["eval_file"]
    prompt = config["prompt"]
    out_dir = config["out_dir"]
    
    mivqa = utils.read_mivqa(data_dir, mivqa_file)
    
    out_file_name = "mivqa_qwen" + "_prompt" + str(prompt) + ".jsonl"
    
    with open(out_file_name, "w") as f:
        for i in tqdm(range(len(mivqa))):
            res = eval_qwen(mivqa, i, template=prompt)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
                
    print("Saved model response to %s, Calculate accuracy"%out_file_name)
    with open(os.path.join(out_dir, out_file_name), "r") as f:
        accuracy = utils.get_accuracy(f, mivqa, parse_fn=utils.parse_qwen)