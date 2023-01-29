for i in {0..4}
do
   python attacks_emotions/agv/agv_attack.py -bf TEST/best_jsons -db "IMAGENET-RESNET" -sae_best=True -img_id $i
done