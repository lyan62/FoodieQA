# VQA 
## Question formulation UI for multi-image VQA
`cd ui` 

- Include the image folder and set the `img_folder` to your local folder path in `config.yaml`.
      

- Edit `config.yaml` 
    - if food_category_file is "processed_food_category_ingredient", select `food_category` from
        ```
        ["seafood", "meat", "vegetable", "tofu", "main-noodle", "main-bao", "main-rice", "main-bread", "main-other", "main-soup", "main-hotpot", "snack", "bbq", "other"]
        ```

    - if food_category_file is "processed_food_category_region", select `food_category` from 
        ```["川", "湘", "赣", "黔", "徽", "闽", "粤", "浙", "苏", "鲁", "新疆","东北","西北", "内蒙", "上海", "其他"]```

    For example,
    ```
    food_category_file: "processed_food_category_region" 
    img_folder: "/Users/projects/foodie-dataset/data-collection"
    food_category: "其他" 
    save_folder: "image-qa"  # folder name to save annotations
    ```

    The `food_category_file` is simple a json file that contains meta data of the images that you would like to use the UI for annotation.
    The data structure is
    ```
    {
        "川":[
            {
                "main_ingredient": xx,
                "id: xx,
                "food_name": xx,
                "food_type": xx,
                ...
                "food_file": img_path,
                "category": xx.
            }
        ],
        [
            ....
        ]
    ...
    }
    ```

- Start UI
    ```
    python3 -m venv foodie
    conda activate foodie
    pip install flask

    python app.py
    ```
- Use the UI as the in the ![annotaion-guide](annotation-guide.png)
Allow choices to the answers to be non-images, **add choices split by comma ","**, see example as in ![following](annotation-guide-choices.png). The "choices" field is optional.
