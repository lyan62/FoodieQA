import argparse
import os

import torch
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
from tqdm import tqdm 
import json


import sys
sys.path.append("/scratch/project/dd-23-107/wenyan/foodie-eval/model-eval")

from scripts import sivqa_utils

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
    
class Evaluator(object):
    def __init__(self, args):
        self.model_base = args.model_base
    
    def _load_model(self):
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        key_info["model_path"] = model_path
        get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)
        return model, tokenizer, image_processor, context_len
    
    @staticmethod
    def format_prompt(text_prompt, args):
        conv = conv_templates[args.conv_mode].copy()
        if isinstance(text_prompt, list):
            qs = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt[0]
            conv.append_message(conv.roles[0], qs) # user 
            conv.append_message(conv.roles[1], text_prompt[1]) # assistant
            conv.append_message(conv.roles[1], None)
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt
            conv.append_message(conv.roles[0], qs) # user 
            conv.append_message(conv.roles[1], None) # assistant
        return conv
    
    def eval_question(self, model, tokenizer, image_processor, context_len, sivqa, idx, args):
    
        question = sivqa[idx]

        q, image_file, choices_str = sivqa_utils.format_question(question, show_food_name=args.show_food_name)
        text_prompt = sivqa_utils.format_text_prompt(q, choices_str, template=args.template, lang=args.lang)
        conv = self.format_prompt(text_prompt, args)
        prompt = conv.get_prompt()
        # print(prompt)
        
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        image = Image.open(os.path.join(args.data_dir, image_file))
        if getattr(model.config, "image_aspect_ratio", None) == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        model = model.to(dtype=torch.bfloat16)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        # print("----------")
        # print("question:", qs)
        # print("outputs:", outputs)
        # print("----------")
        return {
                "response": outputs,
                "qid": sivqa[idx]["question_id"]
            }



def main(args):
    # load model and processor
    evaluator = Evaluator(args)
    
    model, tokenizer, image_processor, context_len = evaluator._load_model()
    

    # read_data
    data_dir = args.data_dir
    eval_file = args.eval_file
    template = args.template
    out_dir = args.out_dir
    
    sivqa = sivqa_utils.read_sivqa(data_dir)
    out_file_name = "sivqa_" + args.model-path.split("/")[-1] + "_prompt" + str(template) + ".jsonl"
    os.makedirs(out_dir, exist_ok=True)
    
    ## eval
    print("Evaluating model on {} questions".format(len(sivqa)))
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for idx in tqdm(range(len(sivqa))):
            res = evaluator.eval_question(model, tokenizer, image_processor, context_len, sivqa, idx, args)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
            
    print("Saved model response to %s"%out_file_name)
    # print("Calculate accuracy...")
    # accuracy = utils.get_accuracy(os.path.join(out_dir, out_file_name), 
    #                               sivqa, parse_fn=utils.parse_idefics_sivqa)
    # print(accuracy)
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/scratch/project/dd-23-107/wenyan/data/foodie/models/Yi-VL-6B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="mm_default")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--data_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie")
    parser.add_argument("--eval_file", default="sivqa_filtered.json")
    parser.add_argument("--out_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie/results")
    parser.add_argument("--show_food_name", action="store_true", default=False)
    parser.add_argument("--template", type=int, default=0)
    parser.add_argument("--lang", default="zh")
    args = parser.parse_args()

    main(args)
