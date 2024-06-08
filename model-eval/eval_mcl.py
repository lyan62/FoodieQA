# For T5 based model
from MIC.model.instructblip import InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel,InstructBlipForConditionalGeneration,InstructBlipProcessor
import datasets
import json
import transformers
from PIL import Image
import torch
import os
from scripts import utils
import argparse
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_images(question):
    image_choices = question["images"]
    return [Image.open(os.path.join(data_dir, image_choice)) for image_choice in image_choices]

def get_prompt(question, replace_token, template=0):
    if template == 0:
        prompt = f'<image0>{replace_token}, <image1>{replace_token}, <image2>{replace_token} <image3> {replace_token}\n'+ 'Answer the following question according to the provided four images, they corresponds to Option (A), Option (B), Option (C), Option (D). Choose one best answer from the given options. Question: {question}, your answer is: Option ('
    elif template == 1:
        prompt = f'Answer the following question according to the provided four images which corresponds to Option (A), Option (B), Option (C), Option (D). Choose one best answer from the given options. The options are: <image0>{replace_token} Option (A)\n<image1>{replace_token} Option (B)\n, <image2>{replace_token} Option (C)\n<image3> {replace_token} Option (D)\nQuestion: {question}, your answer is: Option ('
    elif template == 2:
        prompt = f'Answer the following question according to the provided four images, and choose one best answer from the given options. The options are: <image0>{replace_token} Option (A)\n<image1>{replace_token} Option (B)\n, <image2>{replace_token} Option (C)\n<image3> {replace_token} Option (D)\nQuestion: {question}, your answer is: Option ('
    elif template ==3:
        prompt = f"Human: Question {question} The options are: Option (A)<image0>{replace_token}\n Option (B)<image1>{replace_token}\n Option (C)<image2>{replace_token}\n Option (D)<image3> {replace_token}\nAssistant: If I have to choose one best answer from the given options， the answer is：Option ("
    else:
        raise ValueError("Invalid template number")
    return prompt


def load_model():
    model_type="instructblip"
    # use large model
    # model_ckpt="BleachNick/MMICL-Instructblip-T5-xxl"
    # processor_ckpt = "Salesforce/instructblip-flan-t5-xxl"

    # use small model
    model_ckpt = "BleachNick/MMICL-Instructblip-T5-xl"
    processor_ckpt = "Salesforce/instructblip-flan-t5-xl"

    config = InstructBlipConfig.from_pretrained(model_ckpt)


    if 'instructblip' in model_type:
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_ckpt,
            config=config, cache_dir=os.environ["HF_HOME"]).to(DEVICE, dtype=torch.bfloat16) 

    processor = InstructBlipProcessor.from_pretrained(
        processor_ckpt,cache_dir=os.environ["HF_HOME"]
    )

    image_palceholder="图"
    sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]

    sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
    replace_token="".join(32*[image_palceholder])
    return model, processor, replace_token
## evaluate 0-shot

def eval_question(mivqa, idx, model, processor, replace_token, template=0):
    question = mivqa[idx]
    images = load_images(question)
    prompt = get_prompt(question["question_en"], replace_token, template=template)
    inputs = processor(images=images, text=prompt, return_tensors="pt")

    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

    inputs = inputs.to(DEVICE)
    outputs = model.generate(
            pixel_values = inputs['pixel_values'],
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            img_mask = inputs['img_mask'],
            do_sample=False,
            max_length=160,
            min_length=50,
            num_beams=8,
            set_min_padding_size =False,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return {
            "qid": question["qid"],
            "response": generated_text
    }
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_dir", default="/scratch/project/dd-23-107/wenyan/cache")
    argparser.add_argument("--data_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie")
    argparser.add_argument("--eval_file", default="mivqa_filtered.json")
    argparser.add_argument("--prompt", type=int, default=0)
    argparser.add_argument("--out_dir", default="/scratch/project/dd-23-107/wenyan/data/foodie/results/mivqa_res")
    argparser.add_argument("--model_name", default="qwen/Qwen-VL")
    
    args = argparser.parse_args()
    
    os.environ['HF_HOME'] = args.cache_dir #'/scratch3/wenyan/cache'


    # load_model
    torch.cuda.empty_cache()
    
    model, processor, replace_token = load_model()

    # read_data
    data_dir = args.data_dir
    mivqa_file = args.eval_file
    prompt = args.prompt
    out_dir = args.out_dir
    
    mivqa = utils.read_mivqa(data_dir, 'mivqa_filtered_bi.json')
    
    out_file_name = "mivqa_mcl" + "_prompt" + str(prompt) + ".jsonl"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_file_name), "w", encoding='utf-8') as f:
        for i in tqdm(range(len(mivqa))):
            res = eval_question(mivqa, i, model, processor, replace_token, template=prompt)
            f.write(json.dumps(res, ensure_ascii=False)+"\n")
                
    # print("Saved model response to %s, Calculate accuracy"%out_file_name)
    # accuracy = utils.get_accuracy(os.path.join(out_dir, out_file_name), mivqa, parse_fn=utils.parse_qwen)