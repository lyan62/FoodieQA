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
        
        processor = AutoTokenizer.from_pretrained(self.model_name, cache_dir=os.environ["HF_HOME"])
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=os.environ["HF_HOME"]
        )
        return model, processor
    
    def eval_question(self, textqa, idx, model, processor, args):
        question = textqa[idx]
        # this is in the same style as qwen model
        query_list = textqa_utils.get_prompt_mistral(question, template=args.template, lang=args.lang)
        
        input_ids = processor.apply_chat_template(
            query_list,
            add_generation_prompt=True,
            return_tensors="pt")

        outputs = model.generate(input_ids.to(model.device), 
                                    eos_token_id=processor.eos_token_id,
                                    max_new_tokens=256)
        response = processor.decode(outputs[0], skip_special_tokens=True)
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
    
    
    out_file_name = "textqa_" + args.model_name.split("/")[-1] + "_prompt" + str(template) + ".jsonl"
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
    argparser.add_argument("--model_name", default="mistralai/Mixtral-8x7B-Instruct-v0.1")  # shenzhi-wang/Llama3-8B-Chinese-Chat#meta-llama/Llama-2-70b, meta-llama/Meta-Llama-3-8B-Instruct
    argparser.add_argument("--show_food_name", action="store_true", default=False)
    argparser.add_argument("--template", type=int, default=0)
    argparser.add_argument("--lang", default="zh")
    
    args = argparser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    # load_model
    torch.cuda.empty_cache()
    
    main(args)
    
    
    
