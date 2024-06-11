import json
import os 
from sklearn.metrics import accuracy_score
import random

# data_dir = "/scratch3/wenyan/data/foodie"
# mivqa_file = "mivqa_filtered.json"

def read_mivqa(data_dir, mivqa_file):
    # Read multi-image vqa data
    with open(os.path.join(data_dir, mivqa_file), "r") as f:
        mivqa = json.load(f)
        
    return mivqa

def parse_mantis(res):
    ans = res["response"][0]
    ans2idx = {
        "A":"0",
        "B":"1",
        "C":"2",
        "D":"3"
    }
    return ans2idx[ans.upper()]

def parse_idefics(res):
    ans = res["response"][0].split("\nAssistant: ")[1].split("图")[1][0]
    ans2idx = {
        "A":"0",
        "B":"1",
        "C":"2",
        "D":"3"
    }
    return ans2idx[ans.upper()]

def parse_idefics_sivqa(res, template):
    ans = res["response"][0].split("\nAssistant: ")[1][0]
    ans2idx = {
        "A":"0",
        "B":"1",
        "C":"2",
        "D":"3"
    }
    
    # fallback to random
    if ans not in ans2idx:
        print("Can not parse response, falling back to random...")
        return random.choice(list(ans2idx.values()))
    return ans2idx[ans.upper()]

def parse_qwen_sivqa(res):
    ans = res["response"].split("（")[1][0]
    ans2idx = {
        "A":"0",
        "B":"1",
        "C":"2",
        "D":"3"
    }
    
    # fallback to random
    if ans not in ans2idx:
        print("Can not parse response, falling back to random...")
        return random.choice(list(ans2idx.values()))
    return ans2idx[ans.upper()]

def parse_qwen(res, template=0):
    ans2idx = {
        "A":"0",
        "B":"1",
        "C":"2",
        "D":"3"
    }
    if template == 3:
        ans = res["response"].split("答案为：Picture")[1].strip()[0]
        return ans
    else:
        ans = res["response"].split("答案为：")[1].split("图")[1][0]
        return ans2idx[ans.upper()]
    
def get_accuracy(result_file, mivqa, parse_fn=parse_idefics):
    # get gts
    gt = [x["answer"] for x in mivqa]
    
    # get all answers
    data = []
    with open(result_file, "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    ## get answers
    all_answers = []
    for d in data:
        try:
            ans = parse_fn(d)
            all_answers.append(ans)
        except:
            # fall back to random
            all_answers.append(random.choice(["0", "1", "2", "3"]))
            print(d["qid"], d)
    
    accuracy = accuracy_score(all_answers, gt)
    print("accuracy is: ", accuracy)
    return accuracy


generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
    
    