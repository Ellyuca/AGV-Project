from fitness import attack_rate
from fitness import inv_attack_rate
from fitness import mean_norm_inf
from fitness import mean_norm_2
from fitness import mean_norm_1
from fitness import mean_norm_0
from fitness import AutoNormalization
from agv_compute_fid import ComputeFid
from fitness import ssim_score
from fitness import ssim_score_not_inv
from fitness import IoU
from fitness import center_distance
from fitness import ssim_score_target

def get_distance_functions(dataset_name=None, model=None):
    if dataset_name is not None and model is not None: #to get names (main args)
        cFid = ComputeFid()
        anFiD = AutoNormalization(200) #max value saw during experiments (on imagenet)
    return {
        'norm_0':                     lambda Xf, X: float(mean_norm_0(Xf,X)),
        'norm_1':                     lambda Xf, X: float(mean_norm_1(Xf,X)),
        'norm_2':                     lambda Xf, X: float(mean_norm_2(Xf,X)),
        'norm_inf':                   lambda Xf, X: float(mean_norm_inf(Xf,X)),
        'fid':                        lambda Xf, X: float(cFid(Xf, X)),
        'fid_normalized':             lambda Xf, X: float(anFiD(cFid(Xf, X))),
        'ssim':                       lambda Xf, X: float(ssim_score(Xf,X)),
        'ssim_not_inv':               lambda Xf, X: float(ssim_score_not_inv(Xf,X)),
        'IoU':                        lambda Xf, X: float(IoU(Xf,X)),
        'center_distance':            lambda Xf, X: float(center_distance(Xf,X)),
        'ssim_score_target':          lambda Xf, X: float(ssim_score_target(Xf,X)),
    }
