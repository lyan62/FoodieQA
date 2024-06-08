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

import textqa_utils
import utils
import argparse
from tqdm import tqdm 

class Evaluator(object):
    def __init__(self, args):
        self.model_name = args.model_name
    
    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        processor = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        return model, processor
    
    def eval_question(self, textqa, idx, model, processor, args):
        question = textqa[idx]
        query_list = textqa_utils.get_prompt_qwen(question, template=args.template, lang=args.lang)
        
        text = processor.apply_chat_template(
            query_list,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = processor([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "response": response,
            "qid": textqa[idx]["qid"]
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
    
    textqa = textqa_utils.read_textqa(data_dir)
    
    if "Mantis" in args.model_name:
        out_file_name = "textqa_" + 'mantis' + "_prompt" + str(template) + ".jsonl"
    elif "qwen" in args.model_name:
        out_file_name = "textqa_" + 'qwen' + "_prompt" + str(template) + ".jsonl"
    else:
        out_file_name = "textqa_" + 'idefics' + "_prompt" + str(template) + ".jsonl"
    os.makedirs(out_dir, exist_ok=True)
    
    ## eval
    print("Evaluating model on {} questions".format(len(textqa)))
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for idx in tqdm(range(len(textqa))):
            res = evaluator.eval_question(textqa, idx, model, processor, args)
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
    argparser.add_argument("--eval_file", default="textqa_filtered.json")
    argparser.add_argument("--out_dir", default="/scratch3/wenyan/data/foodie/results/textqa_res")
    argparser.add_argument("--model_name", default="qwen")
    argparser.add_argument("--show_food_name", action="store_true", default=False)
    argparser.add_argument("--template", type=int, default=0)
    argparser.add_argument("--lang", default="zh")
    
    args = argparser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    # load_model
    torch.cuda.empty_cache()
    
    main(args)
    
    
    
