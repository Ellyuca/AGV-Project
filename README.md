# AGV-Project 
 This is the official repository of the AGV-Project for crafting evolutionay adversarial attacks using Instagram inspired image filters.


## AGV-emotions_attacks
Agv_emotions_attacks contains the code to attack an Emotion Recognizer.
The Emotion Recognizer has been trained using the AffectNet dataset that can be downloaded from this [link](http://mohammadmahoor.com/affectnet/) upon request.

Before running the code make sure to add the images for the adversarial attack in the following folder: datasets/images_dataset/emo_images/selected_balanced_images_renamed.

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
```sh
python attacks_emotions/agv/agv_attack.py -l TEST.txt -o TEST.json -bs 1 -e 10 -pp 
"offsprings" -ps "direct" -po "ES"  -np 10 -el true -s pareto -df1 ssim  
-db "EMO-SUBSET-80-EMO-MOBILENET_V2-SINGLE"  -nf 3 -r true -lf TEST.out   
```
