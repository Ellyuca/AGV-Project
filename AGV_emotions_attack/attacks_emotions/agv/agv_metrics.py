import numpy as np
from fitness import attack_rate
from fitness import mean_norm_inf
from fitness import mean_norm_2
from fitness import mean_norm_1
from fitness import mean_norm_0

def compute_metricts(agv_model, nnmodel, dataset_name, X, Y, X_ref, Y_ref):
    Xf = np.array([agv_model.apply(X[i]) for i in range(X.shape[0])])    
    attack_rate_fit = attack_rate(nnmodel, Xf, X)
    mean_norm_inf_fit = mean_norm_inf(Xf, X)
    mean_norm_2_fit = mean_norm_2(Xf, X)
    mean_norm_1_fit = mean_norm_1(Xf, X)
    mean_norm_0_fit = mean_norm_0(Xf, X)

    return {
        "Attack rate": float(attack_rate_fit),
        "norm inf": float(mean_norm_inf_fit),
        "norm 2": float(mean_norm_2_fit),
        "norm 1": float(mean_norm_1_fit),
        "norm 0": float(mean_norm_0_fit),
       
    }
    