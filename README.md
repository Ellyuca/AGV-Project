# AGV-Project
This is the official repository of the AGV-Project for crafting evolutionay adversarial attacks using Instagram inspired image filters.
The Emotion Recognizer has been trained on the AffectNet dataset that can be downloaded from this [link](http://mohammadmahoor.com/affectnet/) upon request.

Before running the code make sure to add the images for the adversarial attack in the folder: datasets/images_dataset/emo_images/selected_balanced_images_renamed.
The images must be renamed using the following scheme: class_id.image.id.jpg.
For examples an image with original id = 842 belonging to the class 7 (representing the Surprise class) must be renamed to:  7.842.jpg

The weights of the trained model can be found in the folder: models/emo_weights.
## 1. Install dependencies.
```sh
pip install -r requirements_cpu.txt
```
To run the code on GPU, install this list instead:
```sh
pip install -r requirements_gpu.txt
```

## 2. Example of how to run the code:
- To launch the attack:
```sh
python attacks_emotions/agv/agv_attack.py -l TEST.txt -o TEST.json -bs 1 -e 10 -pp  "offsprings" -ps "direct" -po "ES"  -np 10 -el true -s pareto -df1 ssim -db "IMAGENET-RESNET"  -nf 3 -r true -lf TEST.out
```

- To test the script with less execution:
```sh
python attacks_emotions/agv/agv_attack.py -l TEST.txt -o TEST.json -bs 1 -e 1 -pp  "offsprings" -ps "direct" -po "ES"  -np 1 -el true -s pareto -df1 ssim -db "IMAGENET-RESNET"  -nf 3 -r true -lf TEST.out
```

- To see image result:
```sh
python attacks_emotions/agv/agv_attack.py -bf TEST/best_jsons -db "IMAGENET-RESNET" -sae_best=True -img_id 0

or

python attacks_emotions/agv/agv_attack.py -i TEST/best_jsons/img_0_TEST.json -t false -sae 1
```