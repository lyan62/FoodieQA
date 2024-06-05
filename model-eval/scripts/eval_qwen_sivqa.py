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

import sivqa_utils
import utils
import argparse
from tqdm import tqdm 

class Evaluator(object):
    def __init__(self, args):
        self.model_name = args.model_name
    
    def _load_model(self):
        model_dir = snapshot_download('qwen/Qwen-VL', cache_dir=os.environ['HF_HOME'])
        # model_dir = snapshot_download('qwen/Qwen-VL-Chat')

        # Loading local checkpoints
        # trust_remote_code is still set as True since we still load codes from local dir instead of transformers
        processor = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, do_image_splitting=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        return model, processor
    
    def eval_question(self, sivqa, idx, model, processor, data_dir, args):
        question = sivqa[idx]
        query_list = sivqa_utils.get_prompt_qwen(question, data_dir, 
                                                    show_food_name=args.show_food_name, 
                                                    template=args.template,
                                                    lang=args.lang)
        
        query = processor.from_list_format(query_list)
        inputs = processor(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        response = processor.decode(pred.cpu()[0], skip_special_tokens=False)
        return {
            "response": response,
            "qid": sivqa[idx]["question_id"]
        }
            



def main(args):
    # load model and processor
    evaluator = Evaluator(args)
    
    model, processor = evaluator._load_model()
    

    # read_data
    data_dir = args.data_dir
    eval_file = args.eval_file
    template = args.template
    out_dir = args.out_dir
    
    sivqa = sivqa_utils.read_sivqa(data_dir)
    
    if "Mantis" in args.model_name:
        out_file_name = "sivqa_" + 'mantis' + "_prompt" + str(template) + ".jsonl"
    elif "qwen" in args.model_name:
        out_file_name = "sivqa_" + 'qwen' + "_prompt" + str(template) + ".jsonl"
    else:
        out_file_name = "sivqa_" + 'idefics' + "_prompt" + str(template) + ".jsonl"
    os.makedirs(out_dir, exist_ok=True)
    
    ## eval
    print("Evaluating model on {} questions".format(len(sivqa)))
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for idx in tqdm(range(len(sivqa))):
            res = evaluator.eval_question(sivqa, idx, model, processor, data_dir, args)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
            
    print("Saved model response to %s"%out_file_name)
    # print("Calculate accuracy...")
    # accuracy = utils.get_accuracy(os.path.join(out_dir, out_file_name), 
    #                               sivqa, parse_fn=utils.parse_idefics_sivqa)
    # print(accuracy)
    
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", default="/scratch3/wenyan/cache")
    argparser.add_argument("--data_dir", default="/scratch3/wenyan/data/foodie")
    argparser.add_argument("--eval_file", default="sivqa_filtered.json")
    argparser.add_argument("--out_dir", default="/scratch3/wenyan/data/foodie/results")
    argparser.add_argument("--model_name", default="qwen")
    argparser.add_argument("--show_food_name", action="store_true", default=False)
    argparser.add_argument("--template", type=int, default=0)
    argparser.add_argument("--lang", default="zh")
    
    args = argparser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    # load_model
    torch.cuda.empty_cache()
    
    main(args)
    
    
    
