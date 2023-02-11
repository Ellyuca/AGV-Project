for i in {0..19}
do
   python attacks_emotions/agv/agv_attack.py -bf TEST/best_jsons -db "IMAGENET-RESNET" -sae_best=True -img_id $i
done

for i in {0..9}
do
   python attacks_emotions/agv/plot_fitness_logs.py -img_id $i
done