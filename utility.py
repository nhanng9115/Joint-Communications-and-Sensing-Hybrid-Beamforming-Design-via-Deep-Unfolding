import torch
import torch.nn as nn
import sys
import h5py
import scipy.io
from system_config import *
import matplotlib.pyplot as plt


# ==================================== initialize F and W ===========================
def initialize(H, R, Pt, normalization):
    if init_scheme == 'conv':
        # randomizing F
        F = torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double) + 1j * torch.randn(len(H[0]), Nt, Nrf,
                                                                                   dtype=torch.double)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    elif init_scheme == 'prop':  # use Le Liang's paper: Low-Complexity Hybrid Precoding in Massive Multiuser MIMO Systems
        # if K == 1:
        #     F = H / torch.abs(H)
        #     F = torch.transpose(F, 2, 3)
        #     Hp = H.conj()
        #     Q = torch.linalg.pinv(Hp)
        #     FQ = torch.linalg.pinv(F) @ Q
        #     W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))
        # else:
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            W = torch.randn(K, len(H[0]), Nrf, M, dtype=torch.double) + 1j * torch.randn(K, len(H[0]), Nrf, M,
                                                                                         dtype=torch.double)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        elif Nrf > M:  # more RF chains than user, need sensing channel as well
            # Determine G
            G = get_mat_G(H,K//2,snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = torch.randn(K, len(H[0]), Nrf, M, dtype=torch.double) + 1j * torch.randn(K, len(H[0]), Nrf, M,
                                                                                         dtype=torch.double)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Fpinv = torch.linalg.pinv(F)
                Wtmp = torch.bmm(Fpinv, Xzf)
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        else:
            sys.stderr.write('Error: Wrong RF chain configuration....\n')
    elif init_scheme == 'svd':
        U, S, V_H = torch.linalg.svd(H)
        V = V_H
        # V = torch.transpose(V_H, 2, 3).conj()
        F = V[:, :, :, :Nrf]
        F = F / torch.abs(F)
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))
    else:
        R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
        at = at0[:, : batch_size, :, :]
        angles_theta = np.around(theta[0, :] * 180 / np.pi)
        idx_snr = np.where(angles_theta == 0)
        at_tmp = at[0, 0, :, idx_snr]
        at1 = at_tmp[:, 0, 0]
        F = H / torch.abs(H)
        F = torch.transpose(F, 2, 3)
        F[:, :, :, 0] = at1
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))

    # rate_0 = get_sum_rate(H, F, W)
    # print(rate_0)
    if normalization == 1:
        # normalize both F and W
        F, W = normalize(F, W, H, Pt)
    else:
        # only normalize W for power constraint
        norm2_FW = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
        W = (torch.sqrt(Pt / norm2_FW.reshape(len(H[0]), 1, 1))) * W
    # rate_0 = get_sum_rate(H, F, W)
    rate_init = torch.zeros(1, len(H[0]))
    beam_error_init = torch.zeros(1, len(H[0]))
    rate_init[0, :] = get_sum_rate(H, F, W, Pt)
    beam_error_init[0, :] = get_beam_error(H, F, W, R, Pt)

    return rate_init, beam_error_init, F, W


# ==================================== initialize F and W with different methods for comparison ===========================
def initialize_schemes(H, R, Pt, init_method):
    if init_method == 'conv':
        # randomizing F
        F = torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double) + 1j * torch.randn(len(H[0]), Nt, Nrf,
                                                                                   dtype=torch.double)
        F = F / torch.abs(F)
        F = torch.cat(((F[None, :, :, :],) * K), 0)
        W = torch.linalg.pinv(H @ F)
    elif init_method == 'prop':  # use Le Liang's paper: Low-Complexity Hybrid Precoding in Massive Multiuser MIMO Systems
        if Nrf == M:
            F = H[K // 2, :, :, :] / torch.abs(H[K // 2, :, :, :])
            F = torch.transpose(F, 1, 2)
            W = torch.randn(K, test_size, Nrf, M, dtype=torch.double) + 1j * torch.randn(K, test_size, Nrf, M,
                                                                                         dtype=torch.double)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
        elif Nrf > M:  # more RF chains than user, need sensing channel as well
            # Determine G
            G = get_mat_G(H, K // 2, snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = torch.randn(K, test_size, Nrf, M, dtype=torch.double) + 1j * torch.randn(K, test_size, Nrf, M,
                                                                                         dtype=torch.double)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Wtmp = torch.linalg.pinv(F) @ Xzf
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)
    elif init_method == 'svd':
        if Nrf == M:
            U, S, V_H = torch.linalg.svd(H)
            V = V_H
            # V = torch.transpose(V_H, 2, 3).conj()
            F = V[:, :, :, :Nrf]
            F = F / torch.abs(F)
            W = torch.randn(K, test_size, Nrf, M, dtype=torch.double) + 1j * torch.randn(K, test_size, Nrf, M,
                                                                                         dtype=torch.double)
            for k in range(K):
                Hk = H[k, :, :, :]
                Hp = Hk.conj()
                Q = torch.linalg.pinv(Hp)
                FQ = torch.linalg.pinv(F) @ Q
                fro_norm = torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = FQ / fro_norm
        elif Nrf > M:
            # Determine G
            G = get_mat_G_SVD(H, K // 2, snr_dB)
            F = G / torch.abs(G)

            F = torch.transpose(F, 1, 2)
            W = torch.randn(K, test_size, Nrf, M, dtype=torch.double) + 1j * torch.randn(K, test_size, Nrf, M,
                                                                                         dtype=torch.double)
            for k in range(K):
                Hk = H[k]
                Hp = Hk.conj()
                Xzf = torch.linalg.pinv(Hp)
                Fpinv = torch.linalg.pinv(F)
                Wtmp = torch.bmm(Fpinv, Xzf)
                Wtmp_norm = torch.linalg.matrix_norm(Wtmp, ord='fro').reshape(len(H[0]), 1, 1)
                W[k, :, :, :] = Wtmp / Wtmp_norm
            F = torch.cat(((F[None, :, :, :],) * K), 0)

    else:
        R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
        at = at0[:, : batch_size, :, :]
        angles_theta = np.around(theta[0, :] * 180 / np.pi)
        idx_snr = np.where(angles_theta == 0)
        at_tmp = at[0, 0, :, idx_snr]
        at1 = at_tmp[:, 0, 0]
        F = H / torch.abs(H)
        F = torch.transpose(F, 2, 3)
        F[:, :, :, 0] = at1
        Hp = H.conj()
        Q = torch.linalg.pinv(Hp)
        FQ = torch.linalg.pinv(F) @ Q
        W = FQ / (torch.linalg.matrix_norm(FQ, ord='fro').reshape(len(H[0]), 1, 1))

    # only normalize W for power constraint
    norm2_FW = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
    W = (torch.sqrt(Pt / norm2_FW.reshape(len(H[0]), 1, 1))) * W

    # rate_0 = get_sum_rate(H, F, W)
    rate_init = torch.zeros(1, len(H[0]))
    beam_error_init = torch.zeros(1, len(H[0]))
    rate_init[0, :] = get_sum_rate(H, F, W, Pt)
    beam_error_init[0, :] = get_beam_error(H, F, W, R, Pt)

    return rate_init, beam_error_init, F, W


# ================== get matrix G for initalization when Nrf > K
def get_mat_G(H,fre_indx,snr_dB):
    G = torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double) + 1j*torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double)
    Htmp = torch.transpose(H[fre_indx, :, :, :], 1, 2)
    G[:, :, :M] = Htmp

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch = at0[:, : batch_size, :, :]
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp = at_batch[0, :, :, angle_index]
        at = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G

def get_mat_G_SVD(H,fre_indx,snr_dB):
    G = torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double) + 1j*torch.randn(len(H[0]), Nt, Nrf, dtype=torch.double)
    U, S, V_H = torch.linalg.svd(H)
    V = V_H
    G[:, :, :M] = V[:, :, :, :M]

    R, at0, theta, ideal_beam = get_radar_data(snr_dB, H)
    at_batch = at0[:, : batch_size, :, :]
    theta_degree = np.around(theta[0, :] * 180 / np.pi)
    for t in range(Nrf - M):
        angle_index = np.where(theta_degree == theta_desire[t])
        at_tmp = at_batch[0, :, :, angle_index]
        at = at_tmp[:, :, 0, 0]
        G[:, :, M + t] = at

    G = torch.transpose(G, 1, 2)
    return G
# ==================================== compute sum rate of MU-MISO system for each subcarrier ===========================
def get_sum_rate(H, F, W, Pt):
    F, W = normalize(F, W, H, Pt)
    # check feasibility
    power_high_threshold = Pt + 10 ** (-3)
    sum_power = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2) / K
    if torch.any(sum_power > power_high_threshold):
        sys.stderr.write('Error: power constraint violated\n')
    # power_high_threshold = Pt + 10 ** (-0)
    # power_low_threshold = Pt - 10 ** (-0)
    # power_all = torch.linalg.matrix_norm(F @ W, ord='fro') ** 2
    # if torch.any(power_all > power_high_threshold):
    #     sys.stderr.write('Error: power constraint violated\n')

    F_H = torch.transpose(F, 2, 3).conj()
    W_H = torch.transpose(W, 2, 3).conj()
    V = W @ W_H  # K x train_size x Nrf x Nrf
    rate = torch.zeros(len(H[0]), )
    for m in range(M):
        # mask_m = torch.ones(len(H), Nrf, M)
        # mask_m[:, :, m] = torch.zeros(len(H), Nrf)
        # W_m = W * mask_m

        W_m = W.clone()  # Create a copy of the original matrix W
        W_m[:, :, :, m] = 0.0  # Set the m-th column to zeros in the new matrix

        # print(W_m[0][0])
        # W_m = W[:, :, :, torch.arange(W.size(3)) != m]
        V_m = W_m @ torch.transpose(W_m, 2, 3).conj()  # need to change to remove 1 column
        h_mk0 = torch.unsqueeze(H[:, :, m, :], dim=2)
        h_mk = torch.transpose(h_mk0, 2, 3)
        h_mk_H = torch.transpose(h_mk, 2, 3).conj()
        Htilde_mk = h_mk @ h_mk_H
        trace_1 = get_trace(F @ V @ F_H @ Htilde_mk)
        trace_2 = get_trace(F @ V_m @ F_H @ Htilde_mk)
        rate = rate + (torch.log2(trace_1 + sigma2) - torch.log2(trace_2 + sigma2)).real
        # print(rate)
        # print((torch.log2(trace_1)).real)
        # print((torch.log2(trace_2)).real)
        # print('-------------------------------------')
    sum_rate = torch.mean(rate)  # sum over all frequencies
    return sum_rate


# ==================================== compute tau function ===========================
def get_beam_error(H, F, W, R, Pt):
    F, W = normalize(F, W, H, Pt)
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    if normalize_tau == 1:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2 / torch.linalg.matrix_norm(R, ord='fro') ** 2
    else:
        error = torch.linalg.matrix_norm(X @ X_H - R, ord='fro') ** 2
    sum_error = torch.mean(error)
    return sum_error


# ==================================== compute MSE ===========================
def get_MSE(F, W, at, R, Pt):
    X = F @ W
    X_H = torch.transpose(X, 2, 3).conj()
    at_H = torch.transpose(at, 2, 3).conj()
    beampattern = torch.real(torch.diagonal(at_H @ X @ X_H @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_mean = torch.mean(beampattern,0)
    # benchmark beampatter
    beam_bm = torch.real(torch.diagonal(at_H @ R @ at, offset=0, dim1=-1, dim2=-2)) / Pt
    beam_bm_mean = torch.mean(beam_bm,0)

    MSE = (torch.abs(beam_bm_mean - beam_mean)) ** 2
    MSE_mean = 10 * torch.log10(torch.mean(torch.mean(MSE, 1)))  # average over channel and get sum
    return MSE_mean


# ==================================== compute trace of matrix A ===========================
def get_trace(A):
    trace_A = torch.diagonal(A, offset=0, dim1=-1, dim2=-2).sum(-1)  # sum all diagonal elements
    return trace_A


# ======== normalization to meet constant modulus and power constraint ===========================
def normalize(F, W, H, Pt):
    F = F / torch.abs(F)
    sum_norm_BB = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
    normalize_factor = torch.sqrt(K * Pt / sum_norm_BB).reshape(len(H[0]), 1, 1)
    W = normalize_factor * W
    # sum_power = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)/K
    # print(sum_power)
    # print(Pt)
    return F, W


# ========================= normalize F based on power constraint =====================
def normalize_power(F, W, H, Pt):
    sum_norm_power = sum(torch.linalg.matrix_norm(F @ W, ord='fro') ** 2)
    normalize_factor = torch.sqrt(Pt / sum_norm_power).reshape(len(H[0]), 1, 1)
    F = normalize_factor * F
    return F


# ======================== generate channels =============================================================
def array_response(N, phi, theta):
    # Generate array response vectors
    a = np.zeros([N, 1], dtype='complex_')
    for n in range(N):
        a[n] = (1 / np.sqrt(N)) * np.exp(1j * np.pi * (n * np.sin(phi)))
    return a


def gen_channel(train_batch_size):
    batch_H = np.zeros([K, train_batch_size, M, Nt],
                       dtype='complex64')  # use to save testing data, used latter in Matlab

    for k in range(K):
        for ii in range(train_batch_size):

            # randomly generate azimuth and elevation angles
            AoD = np.zeros([2, Ncluster * Nray], dtype='complex64')
            AoA = np.zeros([2, Ncluster * Nray], dtype='complex64')

            for cc in range(Ncluster):
                AoD_m = np.random.uniform(0, 2 * np.pi, 2)
                AoA_m = np.random.uniform(0, 2 * np.pi, 2)

                AoD[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoD_m[0], angle_sigma, Nray)
                AoD[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoD_m[1], angle_sigma, Nray)
                AoA[0, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoA_m[0], angle_sigma, Nray)
                AoA[1, cc * Nray:(cc + 1) * Nray] = np.random.laplace(AoA_m[1], angle_sigma, Nray)

            alpha = np.sqrt(sigma / 2) * (
                    np.random.normal(0, 1, Ncluster * Nray) + 1j * np.random.normal(0, 1, Ncluster * Nray))

            # generate channel matrix
            H = np.zeros([M, Nt], dtype='complex_')
            At = np.zeros([Nt, Ncluster * Nray], dtype='complex64')

            for j in range(Ncluster * Nray):
                at = array_response(Nt, AoD[0, j], AoD[1, j])  # UPA array response
                ar = array_response(M, AoA[0, j], AoA[1, j])  # UPA array response
                H = H + alpha[j] * ar * at.conj().T
            H = gamma * H
            batch_H[k, ii, :, :] = H

    return batch_H


# =================================== save generated data ==================================================
def save_data(data_train, data_test):
    # write data
    with h5py.File(data_path_train, 'w') as hf:
        hf.create_dataset('train_set', data=data_train)
    with h5py.File(data_path_test, 'w') as hf:
        hf.create_dataset('test_set', data=data_test)
    # scipy.io.savemat('./channel.mat', {'channel':data_test})


# =================================== load data generated in Matlab ==================================================
def load_data_matlab():
    data_train = scipy.io.loadmat(data_path_train)
    data_train_array = data_train['H_train']
    data_test = scipy.io.loadmat(data_path_test)
    data_test_array = data_test['H_test']
    return data_train_array, data_test_array


# =================================== load data generated in python ==================================================
def load_data():
    # read data
    with h5py.File(data_path_train, 'r') as hf:
        data_train = list(hf.keys())[0]
        data_train_array = hf[data_train][()]
    with h5py.File(data_path_test, 'r') as hf:
        data_test = list(hf.keys())[0]
        data_test_array = hf[data_test][()]
    return data_train_array, data_test_array


# =================================== load data and convert to tensor for trainign=================================
def get_data_tensor(data_source):
    # first load the saved data
    if data_source == 'python':
        data_train_array, data_test_array = load_data()
    else:  # use matlab data
        data_train_array, data_test_array = load_data_matlab()
    # then convert numpy to tensor
    H_train_tensor = torch.from_numpy(data_train_array)
    H_test_tensor = torch.from_numpy(data_test_array)
    return H_train_tensor, H_test_tensor


# =================================== load radar data generated in Matlab ==================================================
def get_radar_data(snr_dB, H):
    # radar info does not depend on channel
    radar_data_file_name = directory_data + 'radar_data.mat'
    radar_data = scipy.io.loadmat(radar_data_file_name)
    # R0_2D = radar_data['Cbar']
    idx_snr = np.where(snr_dB_list == snr_dB)

    if K == 1:
        R0_4D = radar_data['J']
        R0_2D = np.squeeze(R0_4D[:, :, 0, idx_snr])
        R_array = np.tile(R0_2D, [1, len(H[0]), 1, 1])

        at_2D = radar_data['a']
        at_array_true = np.tile(at_2D, (1, train_size, 1, 1))
        at0 = np.expand_dims(at_2D, axis=0)
        at_array1 = np.tile(at0, (train_size, 1, 1, 1))
        at_array = np.transpose(at_array1, (1, 0, 2, 3))  # test
    else:
        R0_4D = radar_data['J']
        R0_2D = np.squeeze(R0_4D[:, :, :, idx_snr])
        R_array0 = np.transpose(R0_2D, (2, 0, 1))
        R_array1 = np.tile(R_array0, [len(H[0]), 1, 1, 1])
        R_array = np.transpose(R_array1, (1, 0, 2, 3))

        at_2D = radar_data['a']
        at0 = np.transpose(at_2D, (2, 0, 1))
        at_array1 = np.tile(at0, (train_size, 1, 1, 1))
        at_array = np.transpose(at_array1, (1, 0, 2, 3))

    R = torch.from_numpy(R_array)
    at = torch.from_numpy(at_array)
    theta = radar_data['theta']
    ideal_beam = radar_data['Pd_theta']

    return R, at, theta, ideal_beam[0, :]


# =================================== get the array of beampattern values ==================================================
def get_beampattern(F, W, at, Pt):
    Q = F @ W
    at_H = torch.transpose(at, 2, 3).conj()
    Q_H = torch.transpose(Q, 2, 3).conj()
    B = at_H @ Q @ Q_H @ at
    # print(torch.linalg.matrix_norm(B, ord='fro') ** 2)
    Bdiag = torch.diagonal(B, offset=0, dim1=-1, dim2=-2) / Pt
    # Bmean = 10 * torch.log10(torch.real(torch.mean(Bdiag, 1)))
    Bmean = torch.real(torch.mean(torch.mean(Bdiag, 1), 0))
    B_array = Bmean.detach().numpy()
    return B_array

# if __name__ == '__main__':
#     # generate data
#     channel_train = gen_channel(train_size)
#     channel_test = gen_channel(test_size)
#
#     # save data
#     save_data(channel_train, channel_valid, channel_test)
#     data_train_array, data_test_array = load_data()
#
#     get_data_tensor()

# print(channel_train[0][0])
# print(data_train_array[0][0])
# print('------------------------------')
# print(channel_test[0][0])
# print(data_test_array[0][0])
# print('------------------------------')
