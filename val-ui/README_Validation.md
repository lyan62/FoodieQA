# Question Validation UI

## Multi-image VQA

Similar as in question formulation, the UI takes a json file that contain the questions belong to each categories. See example as below. The json is indexed by the category names as key, values are list of questions for validation.

```
{
    "闽": [ 
        {
            "question": "哪一道菜中含有干贝？",
            "choices": "",
            "answer": "1",
            "question_type": "ingredients",
            "question_id": "5cff42e986afc707c83ee411ae4af2e6_0",
            "ann_group": "闽",
            "images": [
                img1_path,
                img2_path,
                img3_path,
                img4_path
            ]
        },
        ...
    ],
    "xx": List[dict]
```


- Config the `config.yaml`, run `python app.py`. 
notice that `food_categories: ["川", "上海"]` now supports multiple categories as a list.

- Use the UI as see the example in the ![val-annotaion-guide](val-ui/val-annotation-guide.png)
    - Mark "Is bad question" as *Yes* if you think the question is confusing (e.g. choices are not reasonable, multiple choices are correct, etc.) and should not be included in the dataset
    - Mark "Number of hops" as **single** if it is a question that **does not** require reasoning. e.g. 哪道菜菜色更亮？Mark it as **multiple** if it is a question require you to reason for multiple steps, e.g. 需要先辨认食物名称，或食物食材等. 如哪道菜不适合痛风患者食用？ 这类问题需要先辨认食材是否包含海鲜类食物，如包含海鲜，则不适合痛风患者食用。

- Other features such as click **Next** to save and start index are the same as the Question formulation UI.


## Single-image VQA

Similarly, the json file contains categorized questions in the format:
    ```
    "0": [
            {
                "question": "图片中的食物通常属于哪个菜系?",
                "choices": [
                    "京菜",
                    "徽菜",
                    "新疆菜",
                    "桂菜"
                ],
                "answer": "2",
                "question_type": "cuisine_type",
                "food_name": xx,
                "question_id": "vqa-0",
                "food_meta": {
                    "main_ingredient": [
                       xx, xx
                    ],
                    "id": 217,
                    "food_name": xx,
                    "food_type": "新疆菜",
                    "food_location": "餐馆",
                    "food_file": img_path
                }
            },
    "xx": List[dict]  
    ```
- Config the `single_vqa_config.yaml`, run `python single_vqa_app.py`. 
notice that `food_categories: ["0"] is or ["1"] or your assigned group.
- Use the UI as see the example in the ![val-annotaion-guide](val-ui/val-annotation-guide.png), in this time the difference is only that there are text choices, and one image.
    - Mark "Is bad question" as *Yes* if you think the question is confusing and should not be included in the dataset, e.g. not only one choice is correct, too simple, does not require the image to answer.
    - Mark "Number of hops" as **single** if it is a question that **does not** require reasoning. 
- Other features such as click **Next** to save and start index are the same as last time.

## TextQA

Json file format
```
"0": [
        {
            "question": "苏式红烧肉是基于以下哪一种食材制作的?",
            "choices": [
                "七分精瘦肉和三分肥肉",
                "五花肉",
                "猪蹄",
                "猪排骨"
            ],
            "answer": "1",
            "question_type": "main-ingredient",
            "food_name": "苏式红烧肉",
            "cuisine_type": "苏菜",
            "question_id": "textqa-0"
        },
        ...]
"xx": List[dict]
```
- Config the `text_qa_config.yaml`, run `python text_qa_app.py`. 
notice that `food_categories: ["0"] is or ["1"] or your assigned group.
- The UI does not require you to put in rationale, and there are only text choices.
    - If you do not know the answer, rather than guess, select, **I do not know**.
    - Mark "Is bad question" as *Yes* if you think the question is confusing and should not be included in the dataset, e.g. not only one choice is correct, too simple, too hard, the choices does not make sense, etc.
    - Mark "Number of hops" as **single** if it is a question that **does not** require reasoning. 
- Other features such as click **Next** to save and start index.