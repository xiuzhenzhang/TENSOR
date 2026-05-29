from pathlib import Path

# Hope we can get rid of absolute path in training scripts.
root_path = Path(__file__).parent.resolve()

# CTLSTM
from parameter_set.TPP.ctlstm_parameter_set import ctlstm_hyperparameter_list as ctlstm_hl

# FENN
from parameter_set.TPP.fenn_parameter_set import fenn_hyperparameter_list as fenn_hl

# FullyNN
from parameter_set.TPP.fullynn_parameter_set import fullynn_hyperparameter_list as fullynn_hl

# Integration-free Neural Marked Point Process
from parameter_set.TPP.ifn_parameter_set import ifn_hyperparameter_list as ifn_hl

# LogNormMix
from parameter_set.TPP.lognormmix_parameter_set import lognormmix_hyperparameter_list as l_hl

# Marked_LogNormMix
from parameter_set.TPP.marked_lognormmix_parameter_set import marked_lognormmix_hyperparameter_list as ml_hl
from parameter_set.TPP.mhp_parameter_set import evaluator_hyperparameter as mhp_p

# Mamba Hawkes Process(MHP)
from parameter_set.TPP.mhp_parameter_set import training_hyperparameter as mhp_t

# Naive Marked Temporal Point Processes(Naive)
from parameter_set.TPP.naive_parameter_set import naive_hyperparameter_list as naive_hl

# Recurrent Marked Hawkes Process(RMTPP)
from parameter_set.TPP.rmtpp_parameter_set import rmtpp_hyperparameter_list as rmtpp_hl

# Self-attentive Hawkes Process(SAHP)
from parameter_set.TPP.sahp_parameter_set import sahp_hyperparameter_list as sahp_hl

# Self-attentive Hawkes Process(SAHP)
from parameter_set.TPP.sahp_text_parameter_set import sahp_text_hyperparameter_list as sahp_text_hl

# TFENN
from parameter_set.TPP.tfenn_parameter_set import tfenn_hyperparameter_list as tfenn_hl

# TFullyNN
from parameter_set.TPP.tfullynn_parameter_set import tfullynn_hyperparameter_list as tfully_hl

# Transformer Hawkes Process(THP)
from parameter_set.TPP.thp_parameter_set import thp_hyperparameter_list as thp_hl

# Transformer-powered Intensity-free Integral-based model-numerical(TIFIB-C)
from parameter_set.TPP.tifn_parameter_set import tifn_hyperparameter_list as tifn_hl

parameter_set = {
    'ctlstm': ctlstm_hl,
    'sahp': sahp_hl,
    'sahp_text': sahp_text_hl,
    'fenn': fenn_hl,
    'tfenn': tfenn_hl,
    'fullynn': fullynn_hl,
    'tfullynn': tfully_hl,
    'ifn': ifn_hl,
    'tifn': tifn_hl,
    'naive': naive_hl,
    'thp': thp_hl,
    'rmtpp': rmtpp_hl,
    'lognormmix': l_hl,
    'marked_lognormmix': ml_hl,
    'mhp': {'train': mhp_t, 'evaluate': mhp_p},
}

def parameter_retriver(opt):
    return parameter_set[opt.model]
