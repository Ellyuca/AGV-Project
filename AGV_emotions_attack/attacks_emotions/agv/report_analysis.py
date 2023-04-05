import json
import os
import numpy as np

obj1 = 'ssim_not_inv' #objective function 1 name
obj2 = 'ssim'         #objective function 2 name

directory_json = 'TEST/best_jsons/'
filenames = os.listdir(directory_json)
json_filenames = sorted(filenames, key= lambda x: int(x.split('_')[1]))

fitness_logs = []
id = -1
for filename in json_filenames:
    # id += 1
    # if id == 8:
    #     continue

    with open(directory_json + filename) as file:
        parsed_json = json.load(file)

    fitness_logs.append(parsed_json['fitness'])


print(len(fitness_logs))
objective_function1 = [fitness[0] for fitness in fitness_logs]
objective_function2 = [fitness[1] for fitness in fitness_logs]

print(f'Mean value for {obj1}: {np.mean(objective_function1)}')
print(f'Mean value for {obj2}: {np.mean(objective_function2)}')
