# Foodie-eval

Evaluate LLMs/VLMs with the foodie benchmark


## Models
### LLMs
- `Yi-6B` 

### VLMs
- `Yi-VL-6B`
- `Yi-VL-34B`

## Scripts
- `eval_text_qa.ipynb`  zero-shot and one-shot evaluation of the text QA.


## Image survey preprocessing
Images and corresponding CSV files can be downloaded from [google drive](https://drive.google.com/drive/folders/1haSXSPMfdYBpkg4wspC0qkxZd16llbDD?usp=sharing)

Use `image_data_cleaning.ipynb` to parse it into clean json files.

## VQA 
### question formulation
`cd ui` 

- Download all files from [here](https://drive.google.com/drive/folders/1WFHN8oznqwAdeGXMGlxJbdCbi1l-zL0R?usp=sharing) to a local folder. Unzip the .zip files and set the `img_folder` to your local folder path in `config.yaml`.
      

- Edit `config.yaml` 
    - if food_category_file is "processed_food_category_ingredient", select `food_category` from
        ```
        ["seafood", "meat", "vegetable", "tofu", "main-noodle", "main-bao", "main-rice", "main-bread", "main-other", "main-soup", "main-hotpot", "snack", "bbq", "other"]
        ```

    - if food_category_file is "processed_food_category_region", select `food_category` from 
        ```["川", "湘", "赣", "黔", "徽", "闽", "粤", "浙", "苏", "鲁", "新疆","东北","西北", "内蒙", "上海", "其他"]```

    ```
    food_category_file: "processed_food_category_region" 
    img_folder: "/Users/wli/projects/foodie-dataset/data-collection"
    food_category: "其他" 
    save_folder: "image-qa"  # folder name to save annotations
    ```

- Start UI
    ```
    python3 -m venv foodie
    conda activate foodie
    pip install flask

    python app.py
    ```
- Use the UI as the in the ![annotaion-guide](ui/annotation-guide.png)

