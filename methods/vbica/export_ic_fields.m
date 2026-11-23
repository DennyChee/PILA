function export_ic_fields(run_dir, n_comp, whiten)
% =========================================================================
% Usage:
%   matlab -batch "addpath(genpath('<VBICA_REPO>/Code')); addpath('methods/vbica'); export_ic_fields('<run_dir>', 5, 1)"
%
% Description:
%   vbICA adapter, step 3: load ONE vbICA decomposition from the adapter working directory and write
%   every IC's epoch-space physical field with vbICA's OWN reconstruction (calc_displ_lincomp, the
%   same call pila_load_ic_run.m uses), plus the pixel coordinates and the per-epoch centering
%   scalar, to a plain -v7 .mat that Python reads. NO truth is read (the legacy unit check and the
%   pixel-to-truth mapping are replaced by: input is metres by construction; pixels are mapped by
%   their own coordinates ICA.llh, which are the scenario grid in metres).
%   Checks (error on failure): S-mode; N matches; recomputed fields == ICA.ts_comp (rel < 1e-9);
%   observed (uncentred) == centred + per-epoch mean (< 1e-9 m).
%
% Outputs: <run_dir>/ic_fields_N<NN>.mat with
%   ic_fields [N x T x M] metres, llh [M x 2] (x_m, y_m), mu_epoch [T x 1] m (per-epoch spatial mean
%   removed by vbICA centering), obs [T x M] m (uncentred input), ard, nfe, steps, converged (1/0).
% Date:     2026-11-23
% =========================================================================
info = jsondecode(fileread(fullfile(run_dir, 'export.json')));
if whiten; wtag = 'wh'; else; wtag = 'nowh'; end
run_tag = sprintf('randinit_N%02d_%s', n_comp, wtag);
scen_dir = fullfile(run_dir, 'workdir', info.scen_rel, info.case);
ica_mat = fullfile(scen_dir, ['matfiles_' run_tag], ['decomposedICA_data_' run_tag '.mat']);
fprintf('[1/3] Loading %s\n', ica_mat);
if ~exist(ica_mat, 'file'); error('export_ic_fields:noResult', 'vbICA result not found: %s', ica_mat); end
R = load(ica_mat, 'ICA', 'options', 'Xd_precen', 'Xd');
ICA = R.ICA; options = R.options; Xd_precen = R.Xd_precen; clear R;
if ~strcmp(ICA.decmode, 'S-mode'); error('export_ic_fields:notSmode', 'Expected S-mode, got %s', ICA.decmode); end
if options.scen.N ~= n_comp; error('export_ic_fields:nMismatch', 'N=%d requested, file has %d', n_comp, options.scen.N); end
N = options.scen.N;
% vbICA silently reduces the PCA dimension when the data covariance is singular (e.g. a noise-free static source is
% rank 1); N components cannot then be extracted. Fail with a clear message instead of an index error.
n_ic_found = size(ICA.U, 2);
if n_ic_found < N
    error('export_ic_fields:rankBelowN', 'N exceeds data rank: vbICA kept only %d dimension(s) for N=%d; rerun with N <= %d', ...
          n_ic_found, N, n_ic_found);
end

fprintf('[2/3] Rebuilding %d IC fields with calc_displ_lincomp ...\n', N);
first = calc_displ_lincomp(ICA, options, 1, []);
[T, M] = size(first);
ic_fields = zeros(N, T, M);
max_rel = 0;
for k = 1:N
    field = calc_displ_lincomp(ICA, options, k, []);
    ic_fields(k, :, :) = field;
    ref = ICA.ts_comp{k};
    max_rel = max(max_rel, max(abs(ref(:) - field(:))) / max(abs(ref(:))));
end
fprintf('  recomputed vs ICA.ts_comp: max relative gap %.3e\n', max_rel);
if max_rel > 1e-9; error('export_ic_fields:reconMismatch', 'IC fields differ from ICA.ts_comp by %.3e', max_rel); end
obs = Xd_precen.ts;
if isfield(ICA, 'ind_col_learn') && numel(ICA.ind_col_learn) == M; obs = obs(:, ICA.ind_col_learn); end
mu_epoch = mean(obs, 2);                       % vbICA centering: one spatial-mean scalar per epoch
llh = ICA.llh(:, 1:2);
ard = []; if isfield(ICA, 'ARD'); ard = ICA.ARD; end
nfe = NaN;
if isfield(ICA, 'net') && isfield(ICA.net, 'energy'); nfe = ICA.net.energy(end); end
% Convergence from the vbICA log (ICA.net.energy keeps only the final value, so the step count is not stored):
% vbICA prints 'Reached maximum of <max_steps> training steps.' when it stops on max_steps.
max_steps = info.max_steps;
log_path = fullfile(run_dir, sprintf('vbica_N%02d.log', N));
if exist(log_path, 'file')
    converged = double(~contains(fileread(log_path), 'Reached maximum of'));
else
    converged = NaN;
end
steps = NaN;

fprintf('[3/3] Saving ...\n');
out = fullfile(run_dir, sprintf('ic_fields_N%02d.mat', N));
save(out, 'ic_fields', 'llh', 'mu_epoch', 'obs', 'ard', 'nfe', 'steps', 'converged', 'max_rel', '-v7');
fprintf('Done. %s  (N=%d, T=%d, M=%d, steps=%g, converged=%d)\n', out, N, T, M, steps, converged);
end
