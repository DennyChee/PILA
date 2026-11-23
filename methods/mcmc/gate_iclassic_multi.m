function gate_iclassic_multi(in_mat, mcmc_repo, out_mat)
% =========================================================================
% Usage:
%   matlab -batch "addpath('methods/mcmc'); gate_iclassic_multi('<exported K=1 .mat>', '<MCMC_REPO>', '<out .mat>')"
%
% Description:
%   Reproduction gate for iclassic_multi.m:
%     (1) NSources = 1, FitOffset = false must reproduce iclassic_insar.m BIT-FOR-BIT
%         (SA chain, MCMC posterior, best fit, model LOS) for both start modes;
%     (2) writes okadaMod / mogiMod fields at the exported points for fixed parameters into out_mat,
%         so the Python side can check them against scenarios/forward_numpy.py (centroid convention).
%   Prints PASS / FAIL per check; errors (exit 1 under -batch) on any FAIL.
% Date:     2026-11-23
% =========================================================================
addpath(mcmc_repo);
addpath(fullfile(mcmc_repo, 'modified models'));
n_fail = 0;
fields = {'m_best', 'm_best_sa', 'post_med', 'post_lo', 'post_hi', 'los_mod_m', 'rmse_mm', 'r2', 'sig_L', 'm_start'};
for mode = {'centre', 'random'}
    R0 = iclassic_insar(in_mat, 'exported', 'StartMode', mode{1}, 'Seed', 7, 'NBay', 2000, 'Plots', false, 'SaveMat', false);
    R1 = iclassic_multi(in_mat, 'NSources', 1, 'FitOffset', false, 'StartMode', mode{1}, 'Seed', 7, ...
                        'NBay', 2000, 'Plots', false, 'SaveMat', false);
    for f = fields
        same = isequal(R0.(f{1}), R1.(f{1}));
        n_fail = n_fail + ~same;
        fprintf('  [%s] start=%s %-10s identical\n', pass_fail(same), mode{1}, f{1});
    end
    same_names = isequal(R0.param_names, R1.param_names);
    n_fail = n_fail + ~same_names;
    fprintf('  [%s] start=%s param_names identical\n', pass_fail(same_names), mode{1});
end

% --- forward fields for the Python cross-check (fixed, arbitrary parameters) ---
S = load(in_mat);
x = double(S.x_east_m(:));  y = double(S.y_north_m(:));
okada_params = [1500, -2000, 4000, 30, 70, 5000, 3000, 2];      % xoff yoff depth(centroid) strike dip L W opening
[oE, oN, oU] = okadaMod(x, y, okada_params(1), okada_params(2), okada_params(3), okada_params(4), ...
                        okada_params(5), okada_params(6), okada_params(7), okada_params(8));
mogi_params = [-2500, 1000, 5000, 3e7];
[mE, mN, mU] = mogiMod(x, y, mogi_params(1), mogi_params(2), mogi_params(3), mogi_params(4));
save(out_mat, 'x', 'y', 'okada_params', 'oE', 'oN', 'oU', 'mogi_params', 'mE', 'mN', 'mU', '-v7');
fprintf('  forward fields written to %s\n', out_mat);
if n_fail > 0
    error('gate_iclassic_multi:fail', '%d check(s) FAILED', n_fail);
end
fprintf('Done. All bit-for-bit checks PASSED.\n');
end

function s = pass_fail(ok)
if ok; s = 'PASS'; else; s = 'FAIL'; end
end
