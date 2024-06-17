import numpy as np
import os
import torch

#/////////////////////////// CONSIONDER SCHEMES /////////////////////////////////////////////////////////
run_conv_PGA = 1           # Conventional PGA without unfolding
run_UPGA_J1 = 1            # Unfolded PGA without any modification (J = 1)
run_UPGA_J10 = 1           # Unfolded PGA with setting J = 10
run_UPGA_J20 = 1           # Unfolded PGA with setting J = 20

# ////////////////////////////////////////////// SYSTEM PARAMS //////////////////////////////////////////////
Nt = 64                 # Num of Tx antennas
M = 4                   # Num of Users
Nrf = 4                 # Num of RF chains
K = 1                   # Num of frequency bands
n_target = 3            # Num of sensing targets
theta_desire = np.array([-60.0, 0.0, 60.0], dtype='float64') # Angles of sensing targets

snr_dB = 12                 # SNR for training and showing the convergences
snr = 10 ** (snr_dB / 10)   # transmit power
sigma2 = 1                  # normalized noise power
snr_dB_list = np.array([0, 2, 4, 6, 8, 10, 12], dtype='float64') # SNR for showing the rate and MSEs

init_W = 'ZF' # initialization scheme
initial_normalization = 0  # normalization for initialization
data_source = 'matlab'  # data generate by matlab or python
init_scheme = 'prop'  # proposed initialization for best convergence

# ////////////////////////////////////////////// TESTING SETUPS //////////////////////////////////////////////
normalize_tau = 0   # 0: non-normalize, 1: tau is normalized as tau/(||Psi||_F^2)
LoS_user = 0        # 0: no LoS channel, 1: 1 user has LoS channel, the other has NLoS channel
# For the figures in the paper, just set normalize_tau = 0 and LoS_user = 0.
if normalize_tau == 0:
    if LoS_user == 0:
        system_config = str(Nt) + "TX_" + str(M) + "UE_" + str(Nrf) + "RF"
    else:
        system_config = str(Nt) + "TX_" + str(M) + "UE_" + str(Nrf) + "RF_LoS"
    OMEGA = 0.3
    n_iter_inner_J10 = 10  # Number of inner iterations (J = 10)
else:
    system_config = str(Nt) + "TX_" + str(M) + "UE_" + str(Nrf) + "RF_normalize"
    OMEGA = 10
    n_iter_inner_J10 = 5  # Number of inner iterations (J = 5)
    run_UPGA_J20 = 0
    LoS_user = 0  # no LoS for normalized case

system_info = str(Nt) + " Tx antennas, " + str(M) + " users, " + str(Nrf) + " RF chains, " + str(K) + " frequency, " + str(LoS_user) + " LoS users, " + str(normalize_tau) + " normalize tau"
print(system_info)

# ////////////////////////////////////////////// MODEL PARAMS //////////////////////////////////////////////
train_size = 500    # size of training set
test_size = 1      # size of testing set
batch_size = 10     # batch size when training
n_epoch = 5         # number of training epochs
learning_rate = 0.001 # learning rate

n_iter_outer = 120      # Number of outer iterations (I)
n_iter_inner_J5 = 5     # Number of inner iterations (J = 5)
n_iter_inner_J20 = 20   # Number of inner iterations (J = 20)

# ============================ TUNING PARAMETERS ===========================
WEIGHT_F_RAD = OMEGA  # fixed
WEIGHT_W_RAD = OMEGA / Nt * K
WEIGHT_F_COM = 1  # fixed
WEIGHT_W_COM = 1

# ========================== initiate step sizes as tensor for training ================
step_size_fixed = 1e-2  # step size of conventional PGA
step_size_conv_PGA = torch.full([n_iter_outer, K + 1], step_size_fixed, requires_grad=True)
step_size_UPGA_J1 = torch.full([n_iter_outer, K + 1], step_size_fixed, requires_grad=True)
step_size_UPGA_J10 = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_fixed, requires_grad=True)
step_size_UPGA_J20 = torch.full([n_iter_inner_J20, n_iter_outer, K + 1], step_size_fixed, requires_grad=True)

# ////////////////////////////////////////////// SAVING RESULTS AND DATA //////////////////////////////////////////////
directory_data = "./dataset/" + system_config + "/"
if not os.path.exists(directory_data):
    os.makedirs(directory_data)
directory_benchmark = directory_data  # To save benchmark results

if data_source == 'python':
    train_data_file_name = "train_data.mat"
    test_data_file_name = "test_data.mat"
else:  # matlab
    train_data_file_name = "train_data_matlab.mat"
    test_data_file_name = "test_data_matlab.mat"

data_path_train = directory_data + train_data_file_name
data_path_test = directory_data + test_data_file_name

# To save trained model
directory_model = "./model/" + system_config + "/"
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

model_file_name_UPGA_J1 = directory_model + 'UPGA_J1.pth'
model_file_name_UPGA_J10 = directory_model + 'UPGA_J10.pth'
model_file_name_UPGA_J20 = directory_model + 'UPGA_J20.pth'

# To save result figures
directory_result = "./sim_results/" + system_config + "/"
if not os.path.exists(directory_result):
    os.makedirs(directory_result)

# define labels in figures
label_conv = 'Conventional PGA'
label_UPGA_J1 = r'Unfolded PGA ' + '$(J = 1)$'
label_UPGA_J10 = r'Unfolded PGA ' + '$(J = ' + str(n_iter_inner_J10) + ')$'
label_UPGA_J20 = r'Unfolded PGA ' + '$(J = ' + str(n_iter_inner_J20) + ')$'
label_ZF = 'ZF (digital, comm. only)'
label_SCA = 'SCA-ManOpt (converged)'
