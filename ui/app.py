import json
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from base64 import b64encode
import os
import yaml

app = Flask(__name__)
app.secret_key = os.urandom(24)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
# Assuming you have a folder called "data-collection" with the images
img_folder = config['img_folder'] #"/Users/wli/projects/foodie-dataset/data-collection"
save_folder = config['save_folder']
food_category = config['food_category'] 
food_category_file = os.path.join(img_folder, config['food_category_file']) + ".json"
#"food_category_region.json") # Change this to the path of your food category file

with open(food_category_file, "r") as file:
    categorized_data = json.load(file)
    # print(categorized_data["川"][0])

# Get the list of image filenames for the "meat" category
region_categories = ["川", "湘", "赣", "黔", "徽", "闽", "粤", "浙", "苏", "鲁", "新疆", "云南", 
                     "东北","西北", "内蒙", "上海", "其他"]
food_type_categories = ["seafood", "meat", "vegetable", "tofu", "main-noodle", 
                                "main-bao", "main-rice", "main-bread","main-other", 
                                "main-soup", "main-hotpot", "snack"]

assert food_category in region_categories + food_type_categories, "Invalid food category"
image_filenames = [categorized_data[food_category][i]["food_file"] for i in range(len(categorized_data[food_category]))]

# Construct the full image paths
all_image_paths = [os.path.join(img_folder, filename) for filename in image_filenames]
print("num images: ", len(all_image_paths))
image_qa = {}
current_index = 0

# Create a function to get the image data
def get_image_data(start_index=0, end_index=4):
    images = all_image_paths[start_index:end_index]
    image_data_list = []
    for image_path in images:
        with open(image_path, "rb") as image_file:
            image_data_str = b64encode(image_file.read()).decode('utf-8')
        image_data_list.append({
            "path": image_path,
            "data": image_data_str,
            "session_id": session.get('session_id', None)
        })
    return image_data_list

    
# Create a function to save the image data to a JSON file
def save_image_data(start_index, data):
    file_name = f"{img_folder}/{save_folder}_{food_category}/image_data_{start_index}_{session.get('session_id', 'unknown')}.json"
    data_to_save = {
        "images": all_image_paths[start_index:start_index+4],
        "image_qa": data,
        "session_id": session.get('session_id', 'unknown')
    }
    os.makedirs(f"{img_folder}/{save_folder}_{food_category}", exist_ok=True)
    with open(file_name, "w") as file:
        json.dump(data_to_save, file, ensure_ascii=False, indent=4)
    return jsonify({"status": "success"})


def handle_user_input():
    global image_qa, current_index
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    questions = request.form.getlist('question')
    answers = request.form.getlist('answer')
    question_types = request.form.getlist('question_type')
    image_qa = [{"question": question, "answer": answer, "question_type": question_type} 
                for question, answer, question_type in zip(questions, answers, question_types)]
    save_image_data(current_index, image_qa)

# Create a function to handle navigation
def navigate(direction):
    global current_index, image_qa
    if direction == "next" and current_index + 4 < len(all_image_paths):
        # Save the current user input
        save_image_data(current_index, image_qa)
        current_index += 4
        image_qa = {}
    elif direction == "prev" and current_index - 4 >= 0:
        # Save the current user input
        save_image_data(current_index, image_qa)
        current_index -= 4
        image_qa = {}
    image_data_list = get_image_data(current_index, current_index + 4)
    print("current_index: ", current_index, [x['path'] for x in image_data_list])
    return render_template(
        'index.html',
        image_data_list=image_data_list,
        start_index=current_index,
        img_folder=img_folder,
        session_id=session.get('session_id', None)
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_index, image_qa
    if request.method == 'POST':
        if request.form.get('question') and request.form.get('answer'):
            handle_user_input()
            direction = request.form['direction']
            return navigate(direction)
        if request.form.get('start_index'):
            try:
                start_index = int(request.form['start_index'])
                print(start_index)
                if start_index >= 0 and start_index < len(all_image_paths):
                    current_index = start_index
                    image_qa = {}
                    return render_template(
                        'index.html', 
                        image_data_list=get_image_data(current_index, current_index + 4),
                        start_index=current_index,
                        img_folder=img_folder,
                        error="Invalid start index. Please enter a value between 0 and {}.".format(len(all_image_paths) - 1)
                    )
            except ValueError:
                # Handle non-integer start_index input
                return render_template(
                    'index.html',
                    image_data_list=get_image_data(current_index, current_index + 4),
                    start_index=current_index,
                    img_folder=img_folder,
                    error="Start index must be an integer."
                )
    else:
        if current_index == 0:
            image_qa = {}
        image_data_list = get_image_data(current_index, current_index + 4)
        return render_template(
            'index.html', 
            image_data_list=image_data_list, 
            start_index=current_index, 
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
    