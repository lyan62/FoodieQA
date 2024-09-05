# Human: {question} Here are the options: {options} Assistant: If had to select one of the options, my answer would be (
import json
import os


def read_sivqa(data_dir):
    question_file = os.path.join(data_dir, "sivqa_filtered_bi.json")
    with open(question_file, 'r', encoding='utf-8') as f:
        sivqa = json.load(f)
    return sivqa

def format_choices(choices, template=0):
    idx2choice = {0:"A", 1:"B", 2:"C", 3:"D"}
    choices_str = ''
    for idx, choice in enumerate(choices):
        choices_str += "（{}) {}\n".format(idx2choice[idx], choice.strip())
    return choices_str

def format_question(question, lang="zh", show_food_name=False, use_web_img=False):
    if lang == "zh":
        q = question["question"].strip()
        choices = question["choices"]
    else:
        q = question["question_en"].strip()
        choices = question["choices_en"]
    
    if show_food_name:
        q = q.replace("图片中的食物", question["food_name"])
    
    if use_web_img and "web_file" in question["food_meta"]:
        img = question["food_meta"]["web_file"]
    else:
        img = question["food_meta"]["food_file"]
    
    choices_str = format_choices(choices)
    
    return q, img, choices_str

def format_text_prompt(q, choices_str, template=0, lang="zh"):
    if lang == "zh":
        if template == 0:
            return "{} 选项有: {}, 请根据上图从所提供的选项中选择一个正确答案，为（".format(q, choices_str)
        if template == 1:
            return "你是一个人工智能助手，请你看图回答以下选择题：{} 选项有: {}, 请从中选择一个正确答案，为（".format(q, choices_str)
        if template == 2:
            return ["你是一个智能助手，现在请看图回答以下选择题：{} 选项有: {}".format(q, choices_str), "我从所提供的选项中选择一个正确答案，为（"]
            # return "用户：你是一个智能助手，现在请看图回答以下选择题：{} 选项有: {}, 智能助手：我从所提供的选项中选择一个正确答案，为（".format(q, choices_str)
        if template == 3:
            return ["{} 这是选项: {} 请根据上图从所提供的选项中选择一个正确答案。".format(q, choices_str), "我选择（"]
        if template == 4:
            return ["{} 这是选项: {} 请根据上图从所提供的选项中选择一个正确答案。请保证你的答案清晰简洁并输出字母选项。".format(q, choices_str), "我选择（"]
            # return "用户：{} 这是选项: {} 请根据上图从所提供的选项中选择一个正确答案。智能助手：我选择（".format(q, choices_str)
    else:
        if template == 0:
            return "{} Here are the options: {} If had to select one of the options, my answer would be (".format(q, choices_str)
        if template == 1:
            return "You are an AI assistant. Please answer the following multiple choice question based on the image: {} Here are the options: {} Please select one of the options as your answer (".format(q, choices_str)
        if template == 2:
            return ["{} Here are the options: {}".format(q, choices_str), "If had to select one of the options, my answer would be ("] 
            # return "Human: {} Here are the options: {} Assistant: If had to select one of the options, my answer would be (".format(q, choices_str)
        if template == 3:
            return ["{} These are the options: {} Please select one of the options as your answer.".format(q, choices_str), "I would select ("]
            # return "Human: {} These are the options: {} Please select one of the options as your answer. Assistant: I would select (".format(q, choices_str)
        

def get_prompt_qwen(question, data_dir, show_food_name=False, use_web_img=False, template=0, lang="zh"):
    # for qwen model
    q, img, choices_str = format_question(question, lang=lang, show_food_name=show_food_name, use_web_img=use_web_img)

    query_list = [{"image": os.path.join(data_dir, img)}]
    text_prompt = format_text_prompt(q, choices_str, template, lang=lang)
    if isinstance(text_prompt, list):
        if lang == "zh":
            query_list.append({"text": "用户："+ text_prompt[0] + "智能助手："+ text_prompt[1]})
        else:
            query_list.append({"text": "Human: "+ text_prompt[0] + "Assistant: "+ text_prompt[1]})
    else:
        query_list.append({"text": format_text_prompt(q, choices_str, template, lang=lang)})

    return query_list

def get_prompt_phi(question, data_dir, show_food_name=False, template=0, lang="zh"):
    # for qwen model
    q, img, choices_str = format_question(question, lang=lang, show_food_name=show_food_name)

    text_prompt = format_text_prompt(q, choices_str, template, lang=lang)
    query_list = []
    if isinstance(text_prompt, list):
        query_list.append({"role": "user", "content": "<|image_1|>\n" + text_prompt[0]})
        query_list.append({"role": "assistant", "content": text_prompt[1]})
    else:
        query_list.append({"role": "user", "content": "<|image_1|>\n" + text_prompt})

    return query_list


def get_prompt_idefics(question, data_dir, show_food_name=False, template=0, lang="zh"):
    # for both idefics2 and mantis
    q, img, choices_str = format_question(question, lang=lang, show_food_name=show_food_name)
    text_prompt = format_text_prompt(q, choices_str, template, lang=lang)
    if isinstance(text_prompt, list):
        query_list = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": os.path.join(data_dir, img)},
                            {"type": "text", "text": text_prompt[0]}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": text_prompt[1]}
                        ]
                    }
                ]
    else:
        query_list = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text_prompt},
                            ]}
                        ]
    return query_list


