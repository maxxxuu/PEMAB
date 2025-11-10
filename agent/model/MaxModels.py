from utils.utils import ROOT_DIR

max_PE_model = f"{ROOT_DIR}/run/multi_armed_bandit/results/" \
               f"MAB_PESym_U(0,1)_vrn(20-500)_mn10_(Attention+llllitePENN)/" \
               f"1/MAB_PESym_U(0,1)_vrn(20-500)_mn10_(Attention+llllitePENN)_PESymtrain_epoch=100_U(0,1)_model.pt"

max_attPE_q_net = f"{ROOT_DIR}/run/multi_armed_bandit/results/" \
                      f"MAB_PESym_U(0,1)_vrn(20-500)_mn10_(Attention+lllitePENN)/" \
                      f"1/MAB_PESym_U(0,1)_vrn(20-500)_mn10_(Attention+lllitePENN)_AttentionPE_uni_rn500_model.pt"
