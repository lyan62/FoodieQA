# Human: {question} Here are the options: {options} Assistant: If had to select one of the options, my answer would be (
import json
import os


def read_textqa(data_dir):
    question_file = os.path.join(data_dir, "textqa_filtered.json")
    with open(question_file, 'r', encoding='utf-8') as f:
        sivqa = json.load(f)
    return sivqa

def format_choices(choices, template=0):
    idx2choice = {0:"A", 1:"B", 2:"C", 3:"D"}
    choices_str = ''
    for idx, choice in enumerate(choices):
        choices_str += "（{}) {}\n".format(idx2choice[idx], choice.strip())
    return choices_str

def format_question(question):
    q = question["question"].strip()
    choices = question["choices"]
    
    choices_str = format_choices(choices)
    
    return q, choices_str

def format_text_prompt(q, choices_str, template=0, lang="zh"):
    if lang == "zh":
        if template == 0:
            return "{} 选项有: {}, 请根据从所提供的选项中选择一个正确答案，为（".format(q, choices_str)
        if template == 1:
            return "你是一个智能助手，请你回答以下选择题：{} 选项有: {}, 请从中选择一个正确答案，为（".format(q, choices_str)
        if template == 2:
            return "你是一个智能助手，现在回答以下选择题：{} 选项有: {}\n".format(q, choices_str) + \
                "智能助手：我从所提供的选项中选择一个正确答案，为（"
            # return "用户：你是一个智能助手，现在请看图回答以下选择题：{} 选项有: {}, 智能助手：我从所提供的选项中选择一个正确答案，为（".format(q, choices_str)
        if template == 3:
            return "{} 这是选项: {} 请从所提供的选项中选择一个正确答案。请保证你的答案清晰简洁并输出字母选项。\n".format(q, choices_str)+ \
                "智能助手：我选择（"
            # return "用户：{} 这是选项: {} 请根据上图从所提供的选项中选择一个正确答案。智能助手：我选择（".format(q, choices_str)
    else:
        if template == 0:
            return "{} Here are the options: {} If I had to select one of the options, my answer would be (".format(q, choices_str)
        if template == 1:
            return "You are an AI assistant. Please answer the following multiple choice question: {} Here are the options: {} Please select one of the options as your answer (".format(q, choices_str)
        if template == 2:
            return ["{} Here are the options: {}".format(q, choices_str), "If I had to select one of the options, my answer would be ("] 
            # return "Human: {} Here are the options: {} Assistant: If had to select one of the options, my answer would be (".format(q, choices_str)
        if template == 3:
            return ["{} These are the options: {} Please select one of the options as your answer.".format(q, choices_str), "I would select ("]
            # return "Human: {} These are the options: {} Please select one of the options as your answer. Assistant: I would select (".format(q, choices_str)
        

def get_prompt_qwen(question, template=0, lang="zh"):
    # for qwen model
    q, choices_str = format_question(question)

    text_prompt = format_text_prompt(q, choices_str, template, lang=lang)
    
    if lang == "zh":
        messages = [
            {"role": "system", "content": "你是一个智能助手。"},
            {"role": "user", "content": text_prompt.replace("你是一个智能助手，", "")}] # avoid repeating from the system prompt
    else:
        raise NotImplementedError

    return messages

def get_prompt_yi(question, template=0, lang="zh"):
    # for qwen model
    q, choices_str = format_question(question)

    text_prompt = format_text_prompt(q, choices_str, template, lang=lang)
    
    if lang == "zh":
        messages = [
            {"role": "user", "content": text_prompt}
            ] 
    else:
        raise NotImplementedError

    return messages

def get_prompt_idefics(question, data_dir, show_food_name=False, template=0, lang="zh"):
    # for both idefics2 and mantis
    q, img, choices_str = format_question(question, show_food_name)
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
