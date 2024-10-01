import sivqa_utils, utils, textqa_utils
import os 
import json
from sklearn.metrics import accuracy_score
import re
import random
import glob
import argparse

ans2idx = {
        "A":"0",
        "B":"1",
        "C":"2",
        "D":"3"
        }

def parse_res(res):
    ans_str = res["response"][0].split("\nAssistant:")[-1].strip()
    ans_letter = re.findall(r'[A-D]', ans_str)
    if not ans_letter or len(ans_letter) == 0:
        print("can not parse ans for res: ", res)
        return random.choice(["0", "1", "2", "3"])
    else:
        ans = ans_letter[0].upper()
        if ans not in ans2idx:
            print("can not parse ans for res: ", res)
            return random.choice(["0", "1", "2", "3"])
        else:
            return ans2idx[ans]
        
def read_res_data(result_dir, res_file):
    data = []
    # "sivqa_mantis_prompt3.jsonl"
    with open(os.path.join(result_dir, res_file), "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
        
def get_accuracy(sivqa, data, parse_fn=parse_res): 
    # get acc
    gts = [s["answer"] for s in sivqa]
    answers = [parse_fn(d) for d in data]
    accuracy = accuracy_score(gts, answers)
    print(accuracy)
    return accuracy

def get_res_and_acc(result_dir, vqa, res_file_prefix, parse_fn):
    res_files = glob.glob(os.path.join(result_dir, res_file_prefix))
    print(sorted(res_files))
    
    all_data = []
    all_acc = []
    for file in sorted(res_files):
        file_name = os.path.basename(file)
        print(file_name)
        # load data
        data = read_res_data(result_dir, file_name)        
        # print(data[0])
        acc = get_accuracy(vqa, data, parse_fn)
        
        all_data.append(data)
        all_acc.append(round(acc,4))
    return all_data, all_acc

def parse_yi_res(res, template=0):
    ans_str = res["response"].strip()
    ans_letter = re.findall(r'[A-D]', ans_str)
    if not ans_letter or len(ans_letter) == 0:
        print("can not parse ans for res: ", res)
        return random.choice(["0", "1", "2", "3"])
    else:
        ans = ans_letter[0].upper()
        if ans not in ans2idx:
            print("can not parse ans for res: ", res)
            return random.choice(["0", "1", "2", "3"])
        else:
            return ans2idx[ans]
        
        
def parse_qwen_res(res, template=0):
    try:
        if template == 0 or template==2:
            ans_str = res["response"].split("my answer would be")[1].strip()
        elif template == 1:
            ans_str = res["response"].split("Please select one of the options as your answer")[1].strip()
        elif template == 3:
            ans_str = res["response"].split("Assistant: I would select")[1].strip()
    except:
        ans_str = res["response"].strip()
    ans_letter = re.findall(r'[A-D]', ans_str)
    if not ans_letter or len(ans_letter) == 0:
        print("can not parse ans for res: ", res)
        return random.choice(["0", "1", "2", "3"])
    else:
        ans = ans_letter[0].upper()
        if ans not in ans2idx:
            print("can not parse ans for res: ", res)
            return random.choice(["0", "1", "2", "3"])
        else:
            return ans2idx[ans]
        
def parse_phi3v_res(res):
    random.seed(42)
    ans_str = res["response"]
    ans_letter = re.findall(r'[A-D]', ans_str)
    if not ans_letter or len(ans_letter) == 0:
        print("can not parse ans for res: ", res)
        return random.choice(["0", "1", "2", "3"])
    else:
        ans = ans_letter[0].upper()
        if ans not in ans2idx:
            print("can not parse ans for res: ", res)
            return random.choice(["0", "1", "2", "3"])
        else:
            return ans2idx[ans]   
        
def parse_textqa_res(res, template=0):
    random.seed(42)
    ans_str = res["response"]
    if template in [0,1]:
        try:
            ans_str = ans_str.split("选择一个正确答案")[1]
        except:
            ans_str = ans_str
    else:
        try: ans_str = ans_str.split("我选择")[1]
        except:
            ans_str = ans_str
    ans_letter = re.findall(r'[A-D]', ans_str)
    if not ans_letter or len(ans_letter) == 0:
        print("can not parse ans for res: ", res)
        return random.choice(["0", "1", "2", "3"])
    else:
        ans = ans_letter[0].upper()
        if ans not in ans2idx:
            print("can not parse ans for res: ", res)
            return random.choice(["0", "1", "2", "3"])
        else:
            return ans2idx[ans]
        

def get_sivqa_accuracy(sivqa, res_file):
    if "sivqa_Yi-VL-6B_en_prompt" in res_file:
        parse_fn = parse_yi_res
    elif "sivqa_idefics2-8b_en_prompt" in res_file or "sivqa_Phi-3" in res_file:
        parse_fn = parse_res
    elif "sivqa_Qwen-VL_en" in res_file:
        parse_fn = parse_qwen_res
    
    res_data, acc = get_res_and_acc(sivqa, res_file, parse_fn)
    print(res_file, acc)
    return res_data, acc


def get_res_and_acc(result_dir, mivqa, res_file_prefix, parse_fn):
    res_files = glob.glob(os.path.join(result_dir, res_file_prefix))
    print(sorted(res_files))
    
    all_data = []
    all_acc = []
    for file in sorted(res_files):
        file_name = os.path.basename(file)
        print(file_name)
        # load data
        data = read_res_data(file_name)        
        # print(data[0])
        acc = get_accuracy(mivqa, data, parse_fn)
        
        all_data.append(data)
        all_acc.append(round(acc,4))
    return all_data, all_acc


def get_textqa_accuracy(sivqa, data, parse_fn): 
    # get acc
    gts = [s["answer"] for s in sivqa]
    answers = [parse_fn(d) for d in data]
    accuracy = accuracy_score(gts, answers)
    print(accuracy)
    return accuracy

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results/sivqa_res")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--task", type=str, default="sivqa")
    parser.add_argument("--res_file_pref", type=str, default="sivqa_Yi-VL-6B_en_prompt*.jsonl")
    args = parser.parse_args()
    
    
    if args.task == "sivqa":
        sivqa = sivqa_utils.read_sivqa(args.data_dir, "sivqa_tidy.json")
        res_data, acc = get_sivqa_accuracy(args.result_dir, sivqa, args.res_file_pref, parse_yi_res)
    
    if args.task == "mivqa":
        mivqa = utils.read_mivqa(args.data_dir, "mivqa_tidy.json")
        res_data, acc = get_res_and_acc(args.result_dir, mivqa, args.res_file_pref, parse_res)
        
    if args.task == "textqa":
        textqa = textqa_utils.read_textqa(args.data_dir)
        data = read_res_data(args.result_dir, args.res_file_pref)
        acc = get_textqa_accuracy(textqa, data, parse_fn=parse_textqa_res)