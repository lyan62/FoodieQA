import json
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from base64 import b64encode
import os
import yaml

app = Flask(__name__)
app.secret_key = os.urandom(24)

with open("text_qa_config.yaml", "r") as file:
    config = yaml.safe_load(file)
# Assuming you have a folder called "data-collection" with the images
img_folder = config['img_folder'] #"/Users/wli/projects/foodie-dataset/data-collection"
save_folder = config['save_folder']
food_categories = config['food_categories'] 

# image_filenames = [categorized_data[food_category][i]["food_file"] for i in range(len(categorized_data[food_category]))]

vqa_file = os.path.join(img_folder, config['vqa_file']) + ".json"
vqa_data = []
with open(vqa_file, "r") as file:
    all_questions = json.load(file)
    for category in food_categories:
        vqa_data.extend(all_questions[category])
        
all_ann_categories = "-".join(food_categories)

global max_index
max_index = len(vqa_data) - 1


# Construct the full image paths
# all_image_paths = [os.path.join(img_folder, filename) for filename in image_filenames]
print("num questions: ", len(vqa_data))

image_qa = {}
current_index = 0

# Create a function to get the image data
def get_vqa_data(start_index=0):
    image_data_list = []
    cur_data = vqa_data[start_index]
    # image_path = cur_data["food_meta"]["food_file"]
    # abs_path = os.path.join(img_folder, image_path)
    # with open(abs_path, "rb") as image_file:
    #     image_data_str = b64encode(image_file.read()).decode('utf-8')
    # image_data_list.append({
    #     "path": abs_path,
    #     "data": image_data_str,
    #     "session_id": session.get('session_id', None)
    # })
    return cur_data

    
# Create a function to save the image data to a JSON file
def save_image_data(start_index, data):
    file_name = f"{img_folder}/{save_folder}_{all_ann_categories}/vqa_{start_index}_{session.get('session_id', 'unknown')}.json"
    os.makedirs(f"{img_folder}/{save_folder}_{all_ann_categories}", exist_ok=True)
    with open(file_name, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    return jsonify({"status": "success"})


def handle_user_input():
    global image_qa, current_index
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    user_answer = request.form.get('answer')
    bad_q = request.form.get('badq')
    num_hops = request.form.get('numhops')
    rationale = request.form.get('rationale')
    
    image_qa = {
        "question": vqa_data[current_index]["question"], 
        "choices":vqa_data[current_index]["choices"], 
        "user-answer": user_answer, 
        "gt": vqa_data[current_index]["answer"],
        "rationale": None,
        "bad_q": bad_q,
        "num_hops": num_hops,
        "question_id": vqa_data[current_index]["question_id"],
        "session_id": "val-"+session.get('session_id', 'unknown'),
        "question_type": vqa_data[current_index]["question_type"],
        "food_name": vqa_data[current_index].get("food_name", None),
        } 
    save_image_data(current_index, image_qa)

# Create a function to handle navigation
def navigate(direction):
    global current_index, image_qa
    if direction == "next" and current_index + 1< len(vqa_data):
        # Save the current user input
        save_image_data(current_index, image_qa)
        current_index += 1
        image_qa = {}
    elif direction == "prev" and current_index - 4 >= 0:
        # Save the current user input
        save_image_data(current_index, image_qa)
        current_index -= 1
        image_qa = {}
    cur_data = get_vqa_data(current_index)
    question = cur_data["question"]
    choices = cur_data["choices"]
    # print("current_index: ", current_index, [x['path'] for x in image_data_list])
    return render_template(
        'text-qa-index.html',
        question=question,
        choices=choices,
        start_index=current_index,
        max_index=max_index,
        img_folder=img_folder,
        session_id=session.get('session_id', None)
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_index, image_qa
    if request.method == 'POST':
        if request.form.get('answer'):
            handle_user_input()
            direction = request.form['direction']
            return navigate(direction)
        if request.form.get('start_index'):
            try:
                start_index = int(request.form['start_index'])
                print(start_index)
                if start_index >= 0 and start_index < len(vqa_data):
                    current_index = start_index
                    image_qa = {}
                    cur_data =get_vqa_data(current_index)
                    return render_template(
                        'text-qa-index.html', 
                        question=cur_data["question"],
                        choices=cur_data["choices"],
                        start_index=current_index,
                        max_index=max_index,
                        img_folder=img_folder,
                        error="Invalid start index. Please enter a value between 0 and {}.".format(len(vqa_data) - 1)
                    )
            except ValueError:
                # Handle non-integer start_index input
                cur_data =get_vqa_data(current_index)
                return render_template(
                    'text-qa-index.html',
                    question=cur_data["question"],
                    choices=cur_data["choices"],
                    start_index=current_index,
                    max_index=max_index,
                    img_folder=img_folder,
                    error="Start index must be an integer."
                )
    else:
        if current_index == 0:
            image_qa = {}
        cur_data =get_vqa_data(current_index)
        return render_template(
            'text-qa-index.html', 
            question=cur_data["question"],
            choices=cur_data["choices"],
            start_index=current_index, 
            max_index=max_index,
            img_folder=img_folder
        )
    
@app.route('/static/<path:filename>')
def serve_image(filename):
    return send_from_directory(img_folder, filename)

@app.route('/raw/<path:filename>')
def serve_raw_image(filename):
    return send_from_directory(img_folder, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
    