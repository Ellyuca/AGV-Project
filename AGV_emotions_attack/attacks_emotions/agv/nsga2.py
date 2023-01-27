def dominates(first,second):
    return (first[0] <  second[0] and first[1] <  second[1]) \
        or (first[0] <= second[0] and first[1] <  second[1]) \
        or (first[0] <  second[0] and first[1] <= second[1])  
        
'''-------non-dominated sorting function-------'''      
def non_dominated_sorting(population_size,chroms_obj_record):
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(population_size):
        s[p]=[]
        n[p]=0
        for q in range(population_size):            
            if dominates(chroms_obj_record[p],chroms_obj_record[q]):
                if q not in s[p]:
                    s[p].append(q)
            elif dominates(chroms_obj_record[q],chroms_obj_record[p]):
                n[p]=n[p]+1
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q
                
    del front[len(front)-1]
    return front
'''--------calculate crowding distance function---------'''
def calculate_crowding_distance(front,chroms_obj_record):    
    distance={m:0 for m in front}
    for o in range(2):
        obj={m:chroms_obj_record[m][o] for m in front}
        sorted_keys=sorted(obj, key=obj.get)
        distance[sorted_keys[0]]=distance[sorted_keys[len(front)-1]]=999999999999
        for i in range(1,len(front)-1):
            if len(set(obj.values()))==1:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]+(obj[sorted_keys[i+1]]-obj[sorted_keys[i-1]])/(obj[sorted_keys[len(front)-1]]-obj[sorted_keys[0]])
    return distance            
'''----------selection----------'''
def selection(population_size,front,chroms_obj_record):
    N=0
    new_pop=[]
    while N < population_size:
        for i in range(len(front)):
            N=N+len(front[i])
            if N > population_size:
                distance=calculate_crowding_distance(front[i],chroms_obj_record)
                sorted_cdf=sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop)==population_size:
                        break                
                    new_pop.append(j)              
                break
            else:
                new_pop.extend(front[i])
    return new_pop
    
'''---------NSGA-2 pass --------'''
def nsga_2_pass(N, chroms_obj_record):
    front = non_dominated_sorting(len(chroms_obj_record),chroms_obj_record)
    return selection(N, front, chroms_obj_record)