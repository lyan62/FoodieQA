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
import argparse

PROMPT_GENERAL = "请从给定选项ABCD中选择一个最合适的答案。"

def get_query_list(question, data_dir, template=0):
    q = question["question"].strip()
    idx2choice = {0:"A", 1:"B", 2:"C", 3:"D"}
    
    if template == 0:
        q = q.replace("以下", "以上")
        query_list = [{"image": os.path.join(data_dir, image)} for image in question["images"]]
        query_list.append({"text": "根据以上四张图回答问题，他们分别为图A, 图B, 图C, 图D, " + PROMPT_GENERAL + "问题：{}, 答案为：图".format(q)})
    
    if template == 1:
        q = q.replace("以下", "以上")
        query_list = []
        images = question["images"]
        for i in range(len(images)):
            query_list.append({"image" : os.path.join(data_dir, images[i])})
            query_list.append({"text" : "图{}\n".format(idx2choice[i])})
        query_list.append({"text": "根据以上四张图回答问题," + PROMPT_GENERAL + "问题：{}, 答案为：图".format(q)})
    
    if template == 2:
        q = q.replace("以上", "以下")
        query_list = [{"text":"根据以下四张图回答问题," + PROMPT_GENERAL}]
        images = question["images"]
        
        for i in range(len(images)):
            query_list.append({"text" : "图{}".format(idx2choice[i])})
            query_list.append({"image" : os.path.join(data_dir, images[i])})
        query_list.append({"text": "问题：{}, 答案为：图".format(q)})
    
    if template == 3:
        q = q.replace("以下", "以上")
        query_list = [{"image": os.path.join(data_dir, image)} for image in question["images"]]
        query_list.append({"text": "根据以上四张图回答问题, 问题：{}, 答案为：Picture".format(q)})
        
    if template == 4:
        q = q.replace("以下", "以上")
        query_list = [{"text": "Human: 问题{}，选项有: ".format(q)}]
        for i in range(len(images)):
            query_list.append({"text" : "图{}".format(idx2choice[i])})
            query_list.append({"image" : os.path.join(data_dir, images[i])})
        query_list.append({"text": "Assistant: 如果从给定选项ABCD中选择一个最合适的答案， 答案为：图"})
    return query_list

def eval_question(mivqa, i, template=0):
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", default="/scratch/project/dd-23-107/wenyan/cache")
    argparser.add_argument("--data_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie")
    argparser.add_argument("--eval_file", default="mivqa_filtered.json")
    argparser.add_argument("--prompt", default=0)
    argparser.add_argument("--out_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie/results")
    argparser.add_argument("--model_name", default="qwen/Qwen-VL")
    
    args = argparser.parse_args()
    
    os.environ['HF_HOME'] = args.cache_dir #'/scratch3/wenyan/cache'


    # load_model
    torch.cuda.empty_cache()
    # Downloading model checkpoint to a local dir model_dir
    try:
        model_dir = snapshot_download('qwen/Qwen-VL', cache_dir=os.environ['HF_HOME'])
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, do_image_splitting=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True
        ).eval()
    # trust_remote_code is still set as True since we still load codes from local dir instead of transformers
    except:
        tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen-VL', trust_remote_code=True, do_image_splitting=False)
        model = AutoModelForCausalLM.from_pretrained(
            'qwen/Qwen-VL',
            device_map="auto",
            trust_remote_code=True
        ).eval()

    # read_data
    data_dir = args.data_dir
    mivqa_file = args.eval_file
    prompt = args.prompt
    out_dir = args.out_dir
    
    mivqa = utils.read_mivqa(data_dir, mivqa_file)
    
    out_file_name = "mivqa_qwen-vl" + "_prompt" + str(prompt) + ".jsonl"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for i in tqdm(range(len(mivqa))):
            res = eval_question(mivqa, i, template=prompt)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
                
    print("Saved model response to %s, Calculate accuracy"%out_file_name)
    accuracy = utils.get_accuracy(os.path.join(out_dir, out_file_name), mivqa, parse_fn=utils.parse_qwen)