from algorithms import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

benchmark = 1
# torch.manual_seed(3407)

# ///////////////////////////////////////// SHOW RATES VS. SNRs ///////////////////////////////////

# Load training data
_, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]
# Create new model and load states
conv_PGA = PGA_Conv(step_size_conv_PGA)
if run_UPGA_J1 == 1:
    model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
    model_UPGA_J1.load_state_dict(torch.load(model_file_name_UPGA_J1))
if run_UPGA_J20 == 1:
    model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
    model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20))
if run_UPGA_J10 == 1:
    model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
    model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10))

rate_conv_PGA = np.zeros([len(snr_dB_list), ], dtype='float_')
rate_UPGA_J1 = np.zeros([len(snr_dB_list), ], dtype='float_')
rate_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype='float_')
rate_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype='float_')

MSE_conv_PGA = np.zeros([len(snr_dB_list), ], dtype='float_')
MSE_UPGA_J1 = np.zeros([len(snr_dB_list), ], dtype='float_')
MSE_UPGA_J20 = np.zeros([len(snr_dB_list), ], dtype='float_')
MSE_UPGA_J10 = np.zeros([len(snr_dB_list), ], dtype='float_')

for ss in range(len(snr_dB_list)):
    snr_dB = snr_dB_list[ss]
    snr_ss = 10 ** (snr_dB / 10)
    print('---------------------- snr = ' + str(snr_dB))

    # load data
    R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)

    # Conventional PGA ====================================
    if run_conv_PGA == 1:
        rate_conv_PGA[ss], _, MSE_conv_PGA[ss] = execute_conv_PGA(conv_PGA, H_test, R, snr_ss)
    if run_UPGA_J1 == 1:
        rate_UPGA_J1[ss], _, MSE_UPGA_J1[ss] = execute_UPGA_J1(model_UPGA_J1, H_test, R, snr_ss)
    if run_UPGA_J10 == 1:
        rate_UPGA_J10[ss], _, MSE_UPGA_J10[ss] = execute_UPGA_J10(model_UPGA_J10, H_test, R, snr_ss)
    if run_UPGA_J20 == 1:
        rate_UPGA_J20[ss], _, MSE_UPGA_J20[ss] = execute_UPGA_J20(model_UPGA_J20, H_test, R, snr_ss)


# plot rate vs MSE ======================================================
fig_tradeoff = plt.figure(3)
plt.rcParams["figure.figsize"] = (6.4, 4.0)
if run_UPGA_J1 == 1:
    plt.plot(MSE_UPGA_J1, rate_UPGA_J1, '--', color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
if run_UPGA_J10 == 1:
    plt.plot(MSE_UPGA_J10, rate_UPGA_J10, ':*', color='red', linewidth=3, markersize=7, label=label_UPGA_J10)
if run_UPGA_J20 == 1:
    plt.plot(MSE_UPGA_J20, rate_UPGA_J20, '-', color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
if run_conv_PGA == 1:
    plt.plot(MSE_conv_PGA, rate_conv_PGA, ':', color='black', linewidth=3, markersize=7, label=label_conv)
if benchmark == 1:
    benchmark_results = scipy.io.loadmat(directory_benchmark + 'result_benchmark')
    rate_ZF = np.squeeze(benchmark_results['rate_ZF_mean'])
    rate_SCA = np.squeeze(benchmark_results['rate_SCA_mean'])
    MSE_ZF = 10 * np.log10(np.squeeze(benchmark_results['MSE_ZF_mean']))
    MSE_SCA = 10 * np.log10(np.squeeze(benchmark_results['MSE_SCA_mean']))
    plt.plot(MSE_SCA, rate_SCA, '-x', color='black', linewidth=3, markersize=7, label=label_SCA)
    plt.plot(MSE_ZF, rate_ZF, '-o', color='purple', linewidth=3, markersize=7, label=label_ZF)
# plt.title(system_params)
plt.xlabel('Average MSE [dB]')
plt.ylabel(r'$R$ [bits/s/Hz]')
plt.grid()
plt.legend(loc='lower left', labelspacing  = 0.15)
plt.savefig(directory_result + 'tradeoff_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png')
plt.savefig(directory_result + 'tradeoff_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps')

# plot rate vs SNR ======================================================
fig_rate = plt.figure(1)
plt.rcParams["figure.figsize"] = (6.4, 4.0)
if run_UPGA_J1 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J1, '--', color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
if run_UPGA_J10 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J10, ':*', color='red', linewidth=3, markersize=7, label=label_UPGA_J10)
if run_UPGA_J20 == 1:
    plt.plot(snr_dB_list, rate_UPGA_J20, '-', color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
if run_conv_PGA == 1:
    plt.plot(snr_dB_list, rate_conv_PGA, ':', color='black', linewidth=3, markersize=7, label=label_conv)
if benchmark == 1:
    plt.plot(snr_dB_list, rate_SCA, '-x', color='black', linewidth=3, markersize=7, label=label_SCA)
    plt.plot(snr_dB_list, rate_ZF, '-o', color='purple', linewidth=3, markersize=7, label=label_ZF)

system_params = r'$N=' + str(Nt) + ', M=' + str(M) + ', N_{\mathrm{RF}}=' + str(Nrf) + ', \omega=' + str(OMEGA) + '$'
# plt.title(system_params)
plt.xlabel('SNR [dB]')
plt.ylabel(r'$R$ [bits/s/Hz]')
plt.grid()
plt.legend(loc='upper left', labelspacing  = 0.15)
plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png')
plt.savefig(directory_result + 'rate_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps')

# plot MSE vs SNR ======================================================
fig_MSE = plt.figure(2)
plt.rcParams["figure.figsize"] = (6.4, 4.0)
if run_UPGA_J1 == 1:
    plt.plot(snr_dB_list, MSE_UPGA_J1, '--', color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
if run_UPGA_J10 == 1:
    plt.plot(snr_dB_list, MSE_UPGA_J10, ':*', color='red', linewidth=3, markersize=7, label=label_UPGA_J10)
if run_UPGA_J20 == 1:
    plt.plot(snr_dB_list, MSE_UPGA_J20, '-', color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
if run_conv_PGA == 1:
    plt.plot(snr_dB_list, MSE_conv_PGA, ':', color='black', linewidth=3, markersize=7, label=label_conv)
if benchmark == 1:
    plt.plot(snr_dB_list, MSE_SCA, '-x', color='black', linewidth=3, markersize=7, label=label_SCA)
    plt.plot(snr_dB_list, MSE_ZF, '-o', color='purple', linewidth=3, markersize=7, label=label_ZF)

# plt.title(system_params)
plt.xlabel('SNR [dB]')
plt.ylabel('Average radar beampattern MSE [dB]')
plt.grid()
plt.legend(loc='best', labelspacing  = 0.15)
plt.savefig(directory_result + 'MSE_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.png')
plt.savefig(directory_result + 'MSE_vs_SNR_' + str(Nt) + '_' + str(OMEGA) + '.eps')





plt.show()
