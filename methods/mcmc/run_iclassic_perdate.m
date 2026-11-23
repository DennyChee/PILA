function run_iclassic_perdate(run_dir, mcmc_repo, date_indices)
% =========================================================================
% Usage:
%   matlab -batch "addpath('methods/mcmc'); run_iclassic_perdate('runs/mcmc/<id>/<run_id>', '<MCMC_REPO>')"
%   (optional 3rd argument: 1-based date indices to run, e.g. [1 15 30], for tests)
%
% Description:
%   Per-epoch MCMC adapter, step 2 of 3 (export.py -> THIS -> import_results.py). For every date
%   exported by methods/mcmc/export.py, runs iclassic_multi.m from N cold random starts (seeds
%   from export.json) and keeps the start with the lowest misfit (same rule for every scenario;
%   the truth is never read). Writes results/<date>.mat with the kept R struct, the misfit of
%   every start, and the wall-clock time of the date (all starts).
%
% Inputs:
%   run_dir   - directory holding export.json and inputs/*.mat (written by export.py)
%   mcmc_repo - path of the MCMC code (iclassic_multi.m, mogiMod/okadaMod in 'modified models',
%               sun69.m); in PBS jobs pass ${MCMC_REPO} from paths.yaml
%
% Outputs:  <run_dir>/results/<scenario>_<date>.mat (R, start_rmse_mm, seeds, settings, date_sec)
% Date:     2026-11-23
% =========================================================================
info_path = fullfile(run_dir, 'export.json');
if ~exist(info_path, 'file')
    error('run_iclassic_perdate:noExport', 'Missing %s (run methods/mcmc/export.py first).', info_path);
end
addpath(mcmc_repo);
addpath(fullfile(mcmc_repo, 'modified models'));
info = jsondecode(fileread(info_path));
res_dir = fullfile(run_dir, 'results');
if ~exist(res_dir, 'dir'); mkdir(res_dir); end
n_dates = numel(info.files);
seeds = info.seeds(:)';
fprintf('[1/2] %s: %d dates, model=%s, K=%d, offset=%d, %d start(s), NBay=%d\n', ...
        info.scenario_id, n_dates, info.model, info.n_sources, info.fit_offset, numel(seeds), info.n_bay);
if nargin < 3 || isempty(date_indices); date_indices = 1:n_dates; end
% Settings stored in every result (resume check); timing is stored PER RESULT so a resume or a date subset
% never loses or under-reports it (the importer sums date_sec over the result files).
settings = struct('model', info.model, 'n_sources', info.n_sources, 'fit_offset', logical(info.fit_offset), ...
                  'seeds', seeds, 'n_bay', info.n_bay, 'start_mode', info.start_mode, 'noise_scale', info.noise_scale);
for t = date_indices(:)'
    in_mat = fullfile(run_dir, info.files{t});
    [~, stem] = fileparts(in_mat);
    out_mat = fullfile(res_dir, [stem '.mat']);
    if exist(out_mat, 'file')
        % Resume only if the stored result was made with the SAME settings; otherwise stop loudly.
        old = load(out_mat, 'settings');
        if ~isfield(old, 'settings') || ~isequal(old.settings, settings)
            error('run_iclassic_perdate:staleResult', ...
                  '%s exists but was made with different settings; move it away or use a new run directory.', out_mat);
        end
        fprintf('  [%d/%d] %s exists with identical settings, skipping (resume)\n', t, n_dates, stem);
        continue
    end
    t0 = tic;
    % A date whose LOS has zero variance (e.g. the reference date of a real stack, all pixels exactly 0) cannot be
    % inverted: the original noise scale sig_L = std(LOS) would be 0. Record it as not invertible (NaN parameters,
    % reason stored) instead of crashing; every method sees the same zero field.
    obs = load(in_mat, 'los_obs_m');
    if std(double(obs.los_obs_m(:)), 1) == 0
        n_par = numel(iclassic_param_count(info.model)) * info.n_sources + logical(info.fit_offset);
        R = struct('m_best', nan(1, n_par), 'post_lo', nan(1, n_par), 'post_med', nan(1, n_par), ...
                   'post_hi', nan(1, n_par), 'los_mod_m', zeros(numel(obs.los_obs_m), 1), 'rmse_mm', NaN, ...
                   'skip_reason', 'zero-variance epoch (e.g. reference date): not invertible'); %#ok<NASGU>
        start_rmse_mm = nan(1, numel(seeds)); date_sec = toc(t0); %#ok<NASGU>
        save(out_mat, 'R', 'start_rmse_mm', 'seeds', 'settings', 'date_sec', '-v7');
        fprintf('  [%d/%d] %s: zero-variance LOS, recorded as not invertible\n', t, n_dates, stem);
        continue
    end
    best = []; start_rmse_mm = nan(1, numel(seeds));
    for s = 1:numel(seeds)
        R = iclassic_multi(in_mat, 'NSources', info.n_sources, 'FitOffset', logical(info.fit_offset), ...
                           'StartMode', info.start_mode, 'Seed', seeds(s), 'NBay', info.n_bay, ...
                           'Plots', false, 'SaveMat', false, 'OutDir', res_dir);
        start_rmse_mm(s) = R.rmse_mm;
        if isempty(best) || R.rmse_mm < best.rmse_mm     % sig_L is fixed per date, so RMSE orders the misfit
            best = R; best.seed = seeds(s);
        end
    end
    date_sec = toc(t0);                                  %#ok<NASGU> wall-clock of this date, all starts
    R = best;                                            %#ok<NASGU>
    save(out_mat, 'R', 'start_rmse_mm', 'seeds', 'settings', 'date_sec', '-v7');
    fprintf('  [%d/%d] %s: best seed %d, RMSE %.3f mm (%.1f s)\n', t, n_dates, stem, best.seed, best.rmse_mm, date_sec);
end
fprintf('[2/2] Done. Results in %s\n', res_dir);
end

function names = iclassic_param_count(model)
% Parameter names of ONE source (same order as iclassic_multi.m), used to size a not-invertible record.
switch lower(model)
    case 'mogi';  names = {'xoff_m','yoff_m','depth_m','volume_m3'};
    case 'sun69'; names = {'xoff_m','yoff_m','depth_m','radius_m','volume_m3'};
    case 'okada'; names = {'xoff_m','yoff_m','depth_m','strike_deg','dip_deg','length_m','width_m','opening_m'};
    otherwise; error('run_iclassic_perdate:badModel', 'Unknown model %s', model);
end
end
