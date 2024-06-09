import requests
import torch
from PIL import Image
from io import BytesIO
import yaml
import os
import json
from tqdm import tqdm 

from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

import sivqa_utils
import utils
import argparse
from tqdm import tqdm 

def get_prompt_idefics2(question, data_dir, template=0, lang="zh"):
    if lang == "en":
        q = question["question_en"]
        if template == 0:
            messages = [ 
                {"role": "user", 
                "content": [{"type": "image"}]*4 + \
                    [{"type": "text", "text": "Answer the following question according to the provided four images which corresponds to Option (A), Option (B), Option (C), Option (D). Choose one best answer from the given options. Question: {}, your answer is: Option (".format(q)}]
                }
            ] 
        if template == 1:
            messages = [ 
                {"role": "user", 
                "content": [{"type": "text", "text": "Answer the following question according to the provided four images, which corresponds to Option (A), Option (B), Option (C), Option (D). Choose one best answer from the given options."}, 
                            {"type": "image"}, {"type": "text", "text": "Option (A)\n"},
                            {"type": "image"}, {"type": "text", "text": "Option (B)\n"},
                            {"type": "image"}, {"type": "text", "text": "Option (C)\n"},
                            {"type": "image"}, {"type": "text", "text": "Option (D)\n"},
                            {"type": "text", "text": "Question: {}, your answer is: Option (".format(q)}]
                }
            ]
        if template == 2:
            messages = [ 
                {"role": "user", 
                "content": [{"type": "text", "text": "Answer the following question according to the provided four images, and choose one best answer from the given options."},
                            {"type": "image"}, {"type": "text", "text": "Option (A)\n"},
                            {"type": "image"}, {"type": "text", "text": "Option (B)\n"},
                            {"type": "image"}, {"type": "text", "text": "Option (C)\n"},
                            {"type": "image"}, {"type": "text", "text": "Option (D)\n"},
                            {"type": "text", "text": "Question: {}, your answer is: Option (".format(q)}]
                }
            ]
        if template == 3:
            messages = [ 
                {"role": "user", 
                "content": [{"type": "text", "text": "Question{} The options are: \n".format(q)},
                            {"type": "text", "text": "Option (A)\n"}, {"type": "image"},
                            {"type": "text", "text": "Option (B)\n"}, {"type": "image"},
                            {"type": "text", "text": "Option (C)\n"}, {"type": "image"},
                            {"type": "text", "text": "Option (D)\n"}, {"type": "image"}]
                },
                {"role": "assistant", 
                "content": "If I have to choose one best answer from the given options， the answer is：Option ("}
            ]
        images = [load_image(os.path.join(data_dir, x)) for x in question["images"]]
        return messages, images
    
    

def get_prompt_phi(question, data_dir, template=0, lang="zh"):
    if lang == "zh":
        q = question["question"]
        if template == 0:
            q = q.replace("以下", "以上")
            messages = [ 
            {"role": "user", 
            "content": "<|image_1|>\n<|image_2|>\n<|image_3|>\n<|image_4|>\n根据以上四张图回答问题，他们分别为图A, 图B, 图C, 图D, 请从给定选项ABCD中选择一个最合适的答案。问题：{}, 答案为：图".format(q)
            }, 
            ] 
        if template == 1:
            q = q.replace("以下", "以上")
            messages = [ 
            {"role": "user", 
            "content": "<|image_1|>图A\n<|image_2|>图B\n<|image_3|>图C\n<|image_4|>图D\n根据以上四张图回答问题, 请从给定选项ABCD中选择一个最合适的答案。问题：{}, 答案为：图".format(q)
            }, 
            ] 
        if template == 2:
            q = q.replace("以下", "以上")
            messages = [ 
            {"role": "user", 
            "content": "根据以下四张图回答问题,请从给定选项ABCD中选择一个最合适的答案。<|image_1|>图A\n<|image_2|>图B\n<|image_3|>图C\n<|image_4|>图D\n问题：{}, 答案为：图".format(q)
            }, 
            ] 
        if template == 3:
            q = q.replace("以上", "以下")
            messages = [ 
            {"role": "user", "content": "问题{}，选项有: <|image_1|>图A\n<|image_2|>图B\n<|image_3|>图C\n<|image_4|>图D\n".format(q)},
            {"role": "assistant", "content": "如果从给定选项ABCD中选择一个最合适的答案， 答案为：图"}
            ]
        return messages
    if lang == "en":
        q = question["question_en"]
        if template == 0:
            messages = [ 
                {"role": "user", 
                "content": "<|image_1|>\n<|image_2|>\n<|image_3|>\n<|image_4|>\nAnswer the following question according to the provided four images which corresponds to Option (A), Option (B), Option (C), Option (D). Choose one best answer from the given options. Question: {}, your answer is: Option (".format(q)
                }
            ] 
        if template == 1:
            messages = [ 
                {"role": "user", 
                "content": "Answer the following question according to the provided four images, which corresponds to Option (A), Option (B), Option (C), Option (D). Choose one best answer from the given options. <|image_1|>Option (A)\n<|image_2|>Option (B)\n<|image_3|>Option (C)\n<|image_4|>Option (D)\nQuestion: {}, your answer is: Option (".format(q)
                }
            ]
        if template == 2:
            messages = [ 
                {"role": "user", 
                "content": "Answer the following question according to the provided four images, and choose one best answer from the given options. <|image_1|>Option (A)\n<|image_2|>Option (B)\n<|image_3|>Option (C)\n<|image_4|>Option (D)\nQuestion: {}, your answer is: Option (".format(q)
                }
            ]
        if template == 3:
            messages = [ 
                {"role": "user", 
                "content": "Question{} The options are: Option (A)<|image_1|>\nOption (B)<|image_2|>\nOption (C)<|image_3|>\nOption (D)<|image_4|>\n".format(q)
                },
                {"role": "assistant", 
                "content": "If I have to choose one best answer from the given options， the answer is：Option ("}
            ]
        return messages
    

class Evaluator(object):
    def __init__(self, args):
        self.model_name = args.model_name
    
    def _load_model(self):
        # phi3 model
        if self.model_name == "microsoft/Phi-3-vision-128k-instruct":
            model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                        device_map="auto",
                                                        cache_dir=os.environ["HF_HOME"],
                                                        trust_remote_code=True, torch_dtype="auto", 
                                                        _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True,cache_dir=os.environ["HF_HOME"],) 
        
        # idefics2 model
        if self.model_name == "HuggingFaceM4/idefics2-8b" or self.model_name == "TIGER-Lab/Mantis-8B-Idefics2":
            processor = AutoProcessor.from_pretrained(self.model_name, 
                                                    cache_dir=os.environ["HF_HOME"], 
                                                    do_image_splitting=False
                                                    )
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, cache_dir=os.environ["HF_HOME"], 
                device_map="auto", torch_dtype=torch.float16
                )    
        return model, processor
    
    def eval_question(self, mivqa, idx, model, processor, data_dir, args):
        question = mivqa[idx]
        if self.model_name == "HuggingFaceM4/idefics2-8b" or self.model_name == "TIGER-Lab/Mantis-8B-Idefics2":
            messages, images = get_prompt_idefics2(question, data_dir, 
                                                    template=args.template,
                                                    lang=args.lang)
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        elif self.model_name == "microsoft/Phi-3-vision-128k-instruct":
            messages = get_prompt_phi(question, data_dir, 
                                    template=args.template,
                                    lang=args.lang)
            images = [Image.open(os.path.join(data_dir, x)) for x in question["images"]] 
        
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, images, return_tensors="pt").to(model.device)
        
        generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        return {
            "response": response,
            "qid": mivqa[idx]["question_id"]
        }



def main(args):
    # load model and processor
    evaluator = Evaluator(args)
    
    model, processor = evaluator._load_model()
    

    # read_data
    data_dir = args.data_dir
    if args.lang == "zh":
        eval_file = "mivqa_filtered.json"
    else:
        eval_file = "mivqa_filtered_bi.json"
    template = args.template
    out_dir = args.out_dir
    
    mivqa = utils.read_mivqa(data_dir, eval_file)
    
    out_file_name = "mivqa_" + args.model_name.split("/")[-1] + "_%s"%args.lang + "_prompt" + str(template) + ".jsonl"
    
    os.makedirs(out_dir, exist_ok=True)
    
    ## eval
    print("Evaluating model on {} questions".format(len(mivqa)))
    with open(os.path.join(out_dir, out_file_name), "w") as f:
        for idx in tqdm(range(len(mivqa))):
            res = evaluator.eval_question(mivqa, idx, model, processor, data_dir, args)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
            
    print("Saved model response to %s"%out_file_name)
    # print("Calculate accuracy...")
    # accuracy = utils.get_accuracy(os.path.join(out_dir, out_file_name), 
    #                               sivqa, parse_fn=utils.parse_idefics_sivqa)
    # print(accuracy)
    
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", default="/scratch/project/dd-23-107/wenyan/cache")
    argparser.add_argument("--data_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie")
    argparser.add_argument("--out_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie/results/mivqa_res")
    argparser.add_argument("--model_name", default="microsoft/Phi-3-vision-128k-instruct") # "TIGER-Lab/Mantis-8B-Idefics2" "HuggingFaceM4/idefics2-8b"
    argparser.add_argument("--template", type=int, default=0)
    argparser.add_argument("--lang", type=str, default="zh")
    
    args = argparser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    # load_model
    torch.cuda.empty_cache()
    
    main(args)
    
    
    
