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

import sivqa_utils
import textqa_utils
import utils
import argparse
from tqdm import tqdm 

class Evaluator(object):
    def __init__(self, args):
        self.model_name = args.model_name
    
    def _load_model(self):
        processor = AutoProcessor.from_pretrained(self.model_name, 
                                              cache_dir=os.environ["HF_HOME"],
                                              do_image_splitting=False) # do_image_splitting is False by default
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, cache_dir=os.environ["HF_HOME"],
            device_map="auto", torch_dtype=torch.float16)
        return model, processor
    
    def eval_question(self, sivqa, idx, model, processor, data_dir, args):
        question = sivqa[idx]
        if args.hide_img:
            messages = textqa_utils.get_prompt_idefics(question, data_dir, 
                                                        show_food_name=args.show_food_name, 
                                                        template=args.template,
                                                        lang=args.lang)
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt")
        else:
            messages = sivqa_utils.get_prompt_idefics(question, data_dir, 
                                                        show_food_name=args.show_food_name, 
                                                        template=args.template,
                                                        lang=args.lang)
            print(messages)
            if args.use_web_img and "web_file" in question["food_meta"]:
                img_file = question["food_meta"]["web_file"]
            else:
                img_file = question["food_meta"]["food_file"]
            
            images = [load_image(os.path.join(data_dir, img_file))]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            
        inputs = {k: v.to() for k, v in inputs.items()}
        generate_ids = model.generate(**inputs, max_new_tokens=500)
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

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
    
    out_file_name = "sivqa_" + args.model_name.split("/")[-1] + "_%s"%args.lang + "_prompt" + str(template) + ".jsonl"
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
    argparser.add_argument("--eval_file", type=str, default="sivqa_filtered_bi.json")
    argparser.add_argument("--out_dir", default="/scratch3/wenyan/data/foodie/results/sivqa_res")
    argparser.add_argument("--model_name", default="TIGER-Lab/Mantis-8B-Idefics2") # "TIGER-Lab/Mantis-8B-Idefics2" "HuggingFaceM4/idefics2-8b"
    argparser.add_argument("--show_food_name", action="store_true", default=False)
    argparser.add_argument("--hide_img", action="store_true", default=False)
    argparser.add_argument("--template", type=int, default=0)
    argparser.add_argument("--lang", type=str, default="zh")
    argparser.add_argument("--use_web_img", action="store_true", default=False)
    
    args = argparser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    # load_model
    torch.cuda.empty_cache()
    
    main(args)
    
    
    
