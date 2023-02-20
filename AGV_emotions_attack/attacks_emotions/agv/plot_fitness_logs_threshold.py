import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Process image id by command line.')
parser.add_argument("-img_id", "-id", type=int, help='image id for plot fitness during epochs', default=0)
inargs = parser.parse_args()

img_id = inargs.img_id
threshold_value_list = [0, 170, 200, 230]
threshold_value_list_color = ['orange', 'red', 'blue', 'green']
print(img_id)
logs = []

plt.rcParams['figure.figsize'] = [10, 8]
fig, axs = plt.subplots(2)
fig.suptitle('Fitness log over epochs')
axs[0].set_title('ssim between CAMs')
axs[1].set_title('ssim between imgs')

i = 0
for threshold_value in threshold_value_list:
    with open(f"TEST/logs_txts_{threshold_value}/logs_txts/img_{img_id}_TEST.txt", "r") as file:
        for epoch in file:
            input_list = epoch.split(']')
            epoch_log = []
            count = 0
            for item in input_list:
                count += 1
                if (count % 3) == 0:
                    fitness = []
                    values = item[2:].split(', ')
                    for value in values:
                        fitness_value = float(value)
                        fitness.append(fitness_value)
                    
                    epoch_log.append(fitness)
        
            logs.append(epoch_log)

    fitness_min = []
    for log in logs:
        center_distance = []
        ssim = []
        for fitness in log:
            center_distance.append(fitness[0])
            ssim.append(fitness[1])
        
        fitness_min.append([np.min(center_distance), np.min(ssim)])


    # print(fitness_min)
    center_distance = [fitness_min[i][0] for i in range(len(fitness_min))]
    ssim = [fitness_min[i][1] for i in range(len(fitness_min))]

    axs[0].plot(center_distance, color=threshold_value_list_color[i] , label=f'{threshold_value}')
    axs[1].plot(ssim, color=threshold_value_list_color[i])
    logs = []
    i += 1


fig.legend().set_title("Threshold color")
plt.savefig(f'TEST/logs_txts/plot_img_{img_id}_TEST.png')