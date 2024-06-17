from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

run_program = 1
plot_figure = 1
save_result = 0
# torch.manual_seed(3407)
# ///////////////////////////////////////// SHOW OBJECTIVE VALUES OVER ITERATIONS ///////////////////////////////////
# Load training data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]

R, at0, theta, ideal_beam = get_radar_data(snr_dB, H_test)
at = at0[:, : test_size, :, :]

if run_program == 1:
    # ====================================================== Conv. PGA ====================================
    if run_conv_PGA == 1:
        print('Running conventional PGA...')
        model_conv_PGA = PGA_Conv(step_size_conv_PGA)
        rate_conv, tau_conv, F_conv, W_conv = model_conv_PGA.execute_PGA(H_test, R, snr, n_iter_outer)
        rate_iter_conv = [r.detach().numpy() for r in (sum(rate_conv) / len(H_test[0]))]
        tau_iter_conv = [e.detach().numpy() for e in (sum(tau_conv) / (len(H_test[0])))]

    # ====================================================== Unfolded PGA with J = 1====================================
    if run_UPGA_J1 == 1:
        print('Running unfolded PGA with J = 1...')
        # Create new model and load states
        model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
        model_UPGA_J1.load_state_dict(torch.load(model_file_name_UPGA_J1))

        # executing unfolded PGA on the test set
        sum_rate_UPGA_J1, tau_UPGA_J1, F_UPGA_J1, W_UPGA_J1 = model_UPGA_J1.execute_PGA(H_test, R, snr, n_iter_outer)
        rate_iter_UPGA_J1 = [r.detach().numpy() for r in (sum(sum_rate_UPGA_J1) / len(H_test[0]))]
        tau_iter_UPGA_J1 = [e.detach().numpy() for e in (sum(tau_UPGA_J1) / (len(H_test[0])))]

    # ====================================================== Proposed Unfolded PGA light ====================================
    if run_UPGA_J10 == 1:
        print('Running unfolded PGA with J = 10...')
        # Create new model and load states
        model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
        model_UPGA_J10.load_state_dict(torch.load(model_file_name_UPGA_J10))

        sum_rate_UPGA_J10, tau_UPGA_J10, F_UPGA_J10, W_UPGA_J10 = model_UPGA_J10.execute_PGA(H_test, R,
                                                                                             snr,
                                                                                             n_iter_outer,
                                                                                             n_iter_inner_J10)
        rate_iter_UPGA_J10 = [r.detach().numpy() for r in (sum(sum_rate_UPGA_J10) / len(H_test[0]))]
        tau_iter_UPGA_J10 = [e.detach().numpy() for e in (sum(tau_UPGA_J10) / (len(H_test[0])))]

    # ====================================================== Proposed Unfolded PGA ====================================
    if run_UPGA_J20 == 1:
        print('Running unfolded PGA with J = 20...')
        # Create new model and load states
        model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
        model_UPGA_J20.load_state_dict(torch.load(model_file_name_UPGA_J20))

        sum_rate_UPGA_J20, tau_UPGA_J20, F_UPGA_J20, W_UPGA_J20 = model_UPGA_J20.execute_PGA(H_test, R, snr, n_iter_outer,
                                                                                             n_iter_inner_J20)
        rate_iter_UPGA_J20 = [r.detach().numpy() for r in (sum(sum_rate_UPGA_J20) / len(H_test[0]))]
        tau_iter_UPGA_J20 = [e.detach().numpy() for e in (sum(tau_UPGA_J20) / (len(H_test[0])))]

    # ============================== generate beampattern ////////////////////////////////////////////////////////////////////
    print('generating beampattern...')
    if run_conv_PGA == 1:
        beam_conv_PGA = get_beampattern(F_conv, W_conv, at, snr)
    if run_UPGA_J1 == 1:
        beam_UPGA_J1 = get_beampattern(F_UPGA_J1, W_UPGA_J1, at, snr)
    if run_UPGA_J10 == 1:
        beam_UPGA_J10 = get_beampattern(F_UPGA_J10, W_UPGA_J10, at, snr)
    if run_UPGA_J20 == 1:
        beam_UPGA_J20 = get_beampattern(F_UPGA_J20, W_UPGA_J20, at, snr)

    # ////////////////////////////////////////////////////////////////////////////////////////////
    #                                SAVE RESULTS
    # //////////////////////////////////////////////////////////////////////////////////////////////
    if save_result == 1:
        print('Saving results...')
        if run_conv_PGA == 1:
            result_conv_PGA_file_name = directory_result + 'result_vs_iter_conv.npz'
            np.savez(result_conv_PGA_file_name, name1=rate_iter_conv, name2=tau_iter_conv, name3=beam_conv_PGA)
        if run_UPGA_J1 == 1:
            result_UPGA_J1_file_name = directory_result + 'result_vs_iter_UPGA_J1.npz'
            np.savez(result_UPGA_J1_file_name, name1=rate_iter_UPGA_J1, name2=tau_iter_UPGA_J1, name3=beam_UPGA_J1)
        if run_UPGA_J10 == 1:
            result_UPGA_J10_file_name = directory_result + 'result_vs_iter_UPGA_J10.npz'
            np.savez(result_UPGA_J10_file_name, name1=rate_iter_UPGA_J10, name2=tau_iter_UPGA_J10, name3=beam_UPGA_J10)
        if run_UPGA_J20 == 1:
            result_UPGA_J20_file_name = directory_result + 'result_vs_iter_UPGA_J20.npz'
            np.savez(result_UPGA_J20_file_name, name1=rate_iter_UPGA_J20, name2=tau_iter_UPGA_J20, name3=beam_UPGA_J20)

if plot_figure == 1:

    # ///////////////////////////////////////// SHOW OBJECTIVE VALUES OVER ITERATIONS ///////////////////////////////////
    benchmark = 1
    iter_number_conv_PGA = np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_J1 = np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_J10 = np.array(list(range(n_iter_outer + 1)))
    iter_number_UPGA_J20 = np.array(list(range(n_iter_outer + 1)))

    # //////////////////////////////// LOADING RESULTS //////////////////////////////////////////
    if save_result == 1:
        if run_conv_PGA == 1:
            result_file_name = directory_result + 'result_vs_iter_conv.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']
        if run_UPGA_J1 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J1.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']
        if run_UPGA_J10 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J10.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']
        if run_UPGA_J20 == 1:
            result_file_name = directory_result + 'result_vs_iter_UPGA_J20.npz'
            result = np.load(result_file_name)
            rate_iter_conv, tau_iter_conv, beam_conv_PGA = result['name1'], result['name2'], result['name3']

    #  /////////////////////////////////////////////////////////////////////////////////////////
    #                               PLOT FIGURES
    # //////////////////////////////////////////////////////////////////////////////////////////
    print('Plotting figures...')
    system_params = r'$N=' + str(Nt) + ', M=' + str(M) + ', N_{\mathrm{RF}}=' + str(Nrf) + ', \mathrm{SNR}=' + str(
        snr_dB) + ' \mathrm{dB}' + ', \omega=' + str(OMEGA) + '$'

    # load benchmark results
    if benchmark == 1:
        benchmark_results = scipy.io.loadmat(directory_benchmark + 'result_benchmark')
        rate_ZF = np.squeeze(benchmark_results['rate_ZF_mean'])
        rate_SCA = np.squeeze(benchmark_results['rate_SCA_mean'])
        tau_ZF = np.squeeze(benchmark_results['tau_ZF_mean'])
        tau_SCA = np.squeeze(benchmark_results['tau_SCA_mean'])

        idx_snr = np.where(snr_dB_list == snr_dB)
        rate_ZF = rate_ZF[idx_snr] * np.ones(n_iter_outer + 1)
        rate_SCA = rate_SCA[idx_snr] * np.ones(n_iter_outer + 1)
        tau_ZF = tau_ZF[idx_snr] * np.ones(n_iter_outer + 1)
        tau_SCA = tau_SCA[idx_snr] * np.ones(n_iter_outer + 1)

        beam_ZF = np.squeeze(benchmark_results['beam_ZF_mean'][:, idx_snr])
        beam_SCA = np.squeeze(benchmark_results['beam_SCA_mean'][:, idx_snr])



    # ==================================== RATES ================================================
    plt.figure()
    if run_UPGA_J1 == 1:
        plt.plot(iter_number_UPGA_J1, rate_iter_UPGA_J1, '--', markevery=5, color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
    if run_UPGA_J10 == 1:
        plt.plot(iter_number_UPGA_J10, rate_iter_UPGA_J10, ':*', markevery=5, color='red', linewidth=3, markersize=7,
                 label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        plt.plot(iter_number_UPGA_J20, rate_iter_UPGA_J20, '-', markevery=5, color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
    if run_conv_PGA == 1:
        plt.plot(iter_number_conv_PGA, rate_iter_conv, ':', markevery=5, color='black', linewidth=3, markersize=7, label=label_conv)
    if benchmark == 1:
        plt.plot(iter_number_conv_PGA, rate_SCA, '-x', markevery=5, color='black', linewidth=3, markersize=7, label=label_SCA)
        plt.plot(iter_number_conv_PGA, rate_ZF, '-o', markevery=5, color='purple', linewidth=3, markersize=7, label=label_ZF)

    # plt.title(system_params)
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel('$R$ [bits/s/Hz]', fontsize="14")
    plt.grid()
    plt.legend(loc='best', fontsize="14", labelspacing  = 0.15)
    # save figure and results
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'rate_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # plot beam errors ////////////////////////////////////////////////////////////////////
    fig_tau = plt.figure(2)
    if run_UPGA_J1 == 1:
        plt.plot(iter_number_UPGA_J1, tau_iter_UPGA_J1, '--', markevery=5, color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
    if run_UPGA_J10 == 1:
        plt.plot(iter_number_UPGA_J10, tau_iter_UPGA_J10, ':*', markevery=5, color='red', linewidth=3, markersize=7,
                 label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        plt.plot(iter_number_UPGA_J20, tau_iter_UPGA_J20, '-', markevery=5, color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
    if run_conv_PGA == 1:
        plt.plot(iter_number_conv_PGA, tau_iter_conv, ':', markevery=5, color='black', linewidth=3, markersize=7, label=label_conv)
    if benchmark == 1:
        plt.plot(iter_number_conv_PGA, tau_SCA, '-x', markevery=5, color='black', linewidth=3, markersize=7, label=label_SCA)
        plt.plot(iter_number_conv_PGA, tau_ZF, '-o', markevery=5, color='purple', linewidth=3, markersize=7, label=label_ZF)

    # plt.title(system_params)
    plt.xlabel(r'Number of iterations/layers $(I)$', fontsize="14")
    plt.ylabel( r'$\bar{\tau}$', fontsize="14")
    plt.grid()
    # plt.legend(loc='best')
    # save figure
    plt.savefig(directory_result + 'beampattern_error_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'beampattern_error_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # plot rate - beam errors tradeoff ////////////////////////////////////////////////////////////////////
    fig_tradeoff = plt.figure(3)
    if run_UPGA_J1 == 1:
        plt.plot(tau_iter_UPGA_J1, rate_iter_UPGA_J1, '--', markevery=5, color='blue', linewidth=3, markersize=7, label=label_UPGA_J1)
    if run_UPGA_J10 == 1:
        plt.plot(tau_iter_UPGA_J10, rate_iter_UPGA_J10, ':*', markevery=5, color='red', linewidth=3, markersize=7,
                 label=label_UPGA_J10)
    if run_UPGA_J20 == 1:
        plt.plot(tau_iter_UPGA_J20, rate_iter_UPGA_J20, '-', markevery=5, color='red', linewidth=3, markersize=7, label=label_UPGA_J20)
    if run_conv_PGA == 1:
        plt.plot(tau_iter_conv, rate_iter_conv, ':', markevery=5, color='black', linewidth=3, markersize=7, label=label_conv)
    if benchmark == 1:
        plt.plot(tau_SCA, rate_SCA, '-x', markevery=5, color='black', linewidth=3, markersize=7, label=label_SCA)
        plt.plot(tau_ZF, rate_ZF, '-o', markevery=5, color='purple', linewidth=3, markersize=7, label=label_ZF)

    # plt.title(system_params)
    plt.xlabel( r'$\bar{\tau}$', fontsize="14")
    plt.ylabel(r'$R$ [bits/s/Hz]', fontsize="14")
    plt.grid()
    # plt.legend(loc='best')
    # save figure
    plt.savefig(directory_result + 'tradeoff_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'tradeoff_vs_iter_' + str(Nt) + '_' + str(OMEGA) + '.eps')

    # /////////////////////////////////////// BEAMPATTERN /////////////////////////////
    fig_beam = plt.figure(4)
    angles_theta = theta[0, :] * 180 / np.pi
    # benchmark beampatter
    at_H = torch.transpose(at, 2, 3).conj()
    beam_bm = torch.diagonal(at_H @ R @ at, offset=0, dim1=-1, dim2=-2) / snr
    beam_bm_array = beam_bm[0,0,:]
    plt.plot(angles_theta, np.real(beam_bm_array), '--', markevery=5, color='green', linewidth=1,
             label='Benchmark beampattern')  # ideal beampatter
    if run_UPGA_J1 == 1:
        plt.plot(angles_theta, beam_UPGA_J1, '--', markevery=5, color='blue', linewidth=2, markersize=7)
    if run_UPGA_J10 == 1:
        plt.plot(angles_theta, beam_UPGA_J10, ':*', markevery=5, color='red', linewidth=3, markersize=7)
    if run_UPGA_J20 == 1:
        plt.plot(angles_theta, beam_UPGA_J20, '-', markevery=5, color='red', linewidth=2, markersize=7)
    if run_conv_PGA == 1:
        plt.plot(angles_theta, beam_conv_PGA, ':', markevery=5, color='black', linewidth=2, markersize=7)
    if benchmark == 1:
        benchmark_results = scipy.io.loadmat(directory_benchmark + 'result_benchmark')
        plt.plot(angles_theta, beam_SCA, '-x', markevery=5, color='black', linewidth=1, markersize=7)
        plt.plot(angles_theta, beam_ZF, '-o', markevery=5, color='purple', linewidth=1, markersize=7)


    # plt.title(system_params)
    plt.xlabel(r'Angle $(\theta_t)$', fontsize="14")
    plt.ylabel('Normalized sensing beampattern', fontsize="14")
    # if Nt == 32:
    #     plt.ylim([0, 6.5])
    # else:
    #     plt.ylim([0, 12])
    plt.xticks(np.arange(-90, 91, step=30))
    plt.grid()
    plt.legend(loc='upper right', fontsize="14")
    # save figure
    plt.savefig(directory_result + 'beampattern_' + str(Nt) + '_' + str(OMEGA) + '.png')
    plt.savefig(directory_result + 'beampattern_' + str(Nt) + '_' + str(OMEGA) + '.eps')
    # plt.ylim([0, 0.4])
    plt.show()
