% =========================================================================
% Usage:       (from the PILA repo root, normally via a jobs/comsol_classical/*.pbs job)
%              EXPORT_DIR=$PWD/exported_comsol/single OUT_DIR=$PWD/classic_out_comsol/flat_desc \
%              GROUP_GLOB='*_flat_desc_*' matlab -batch "run('classical_comsol/run_comsol_group.m')"
% Description: Classical SA + Bayesian MCMC Mogi inversion (iclassic_insar.m, 'exported'
%              mode) of every COMSOL wk6 export matching GROUP_GLOB, one scene at a time.
%              Each scene's result goes to OUT_DIR/<config>/mogi_<config>_20000101_result.mat.
%              Scenes that already have a result are skipped, so a re-run resumes safely.
% Inputs:      env EXPORT_DIR (dir of export .mat files), OUT_DIR, GROUP_GLOB,
%              optional NBAY (default 1e4), SEED (default 1), START_MODE (default centre),
%              CHUNK_INDEX / CHUNK_COUNT (run only slice CHUNK_INDEX of the sorted group).
% Outputs:     per-scene *_result.mat (-v7.3), *_fit.png, *_posterior.png, and
%              OUT_DIR/_group_summary.csv (one row per scene run in THIS job).
% Date:        2026-10-02
%
% Scientific notes:
%   * Same settings as the field benchmark: NBay = 1e4, seed 1, SA from the box centre.
%   * The likelihood sigma is std(LOS) (iclassic_insar), so on these noise-free maps the
%     MCMC intervals are NOT noise-based uncertainties; compare point estimates.
% =========================================================================
clear; clc; close all;

% --- Paths and settings (from the environment) -----------------------------
export_dir = getenv('EXPORT_DIR');
out_dir    = getenv('OUT_DIR');
group_glob = getenv('GROUP_GLOB');
if isempty(export_dir) || isempty(out_dir) || isempty(group_glob)
    error('run_comsol_group:env', 'EXPORT_DIR, OUT_DIR and GROUP_GLOB must all be set');
end
% run() cd's into this script's folder while it executes, so a relative path would resolve
% against classical_comsol/. Require absolute paths (the generated PBS jobs pass them).
if ~startsWith(export_dir, '/') || ~startsWith(out_dir, '/')
    error('run_comsol_group:relPath', 'EXPORT_DIR and OUT_DIR must be absolute paths');
end
n_bay = str2double(getenv('NBAY'));        if isnan(n_bay); n_bay = 1e4; end
seed  = str2double(getenv('SEED'));        if isnan(seed);  seed = 1;    end
start_mode = getenv('START_MODE');         if isempty(start_mode); start_mode = 'centre'; end

% Pin threads to the PBS allocation so runtimes are repeatable (timing is compared to PILA).
n_thread = str2double(getenv('NCPUS'));
if isnan(n_thread) || n_thread < 1; n_thread = 4; end
maxNumCompThreads(n_thread);

addpath('/eos-rs/INSAR_processing/denny/matlab/synthetic_deformation/Volcano_Deformation');
addpath('/eos-rs/INSAR_processing/denny/matlab/synthetic_deformation/Volcano_Deformation/modified models');

% --- [1/3] Find the scenes of this group -----------------------------------
files = dir(fullfile(export_dir, [group_glob '.mat']));
if isempty(files)
    error('run_comsol_group:noFiles', 'no exports match %s', fullfile(export_dir, [group_glob '.mat']));
end
% Optional chunking (CHUNK_INDEX 1..CHUNK_COUNT) so one group can be split over parallel
% jobs: ~200 s per scene measured, 40 scenes would be ~2.5 h in one job.
[~, order] = sort({files.name});
files = files(order);
chunk_count = str2double(getenv('CHUNK_COUNT'));
chunk_index = str2double(getenv('CHUNK_INDEX'));
chunk_tag = '';
if ~isnan(chunk_count) && chunk_count > 1
    chunk_edges = round(linspace(0, numel(files), chunk_count + 1));
    files = files(chunk_edges(chunk_index) + 1 : chunk_edges(chunk_index + 1));
    chunk_tag = sprintf('_chunk%02dof%02d', chunk_index, chunk_count);
end
fprintf('[1/3] %d scenes in %s matching %s%s  (NBay=%g, seed=%d, start=%s, threads=%d)\n', ...
        numel(files), export_dir, group_glob, chunk_tag, n_bay, seed, start_mode, n_thread);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% --- [2/3] Invert each scene -----------------------------------------------
config_names = cell(numel(files), 1);
status = cell(numel(files), 1);
r2_values = nan(numel(files), 1);
runtime_total_sec = nan(numel(files), 1);
t_group = tic;
for ii = 1:numel(files)
    [~, config_name] = fileparts(files(ii).name);
    config_names{ii} = config_name;
    scene_out_dir = fullfile(out_dir, config_name);
    result_mat = fullfile(scene_out_dir, sprintf('mogi_%s_20000101_result.mat', config_name));
    if exist(result_mat, 'file')
        % Already done (earlier attempt): never overwrite a finished result.
        fprintf('[2/3] (%d/%d) %s: result exists, skipped\n', ii, numel(files), config_name);
        status{ii} = 'skipped';
        % Keep the summary complete after a resume: read the stored fit metrics.
        stored = load(result_mat, 'config');
        r2_values(ii) = stored.config.r2;
        runtime_total_sec(ii) = stored.config.runtime_total_sec;
        continue
    end
    fprintf('[2/3] (%d/%d) %s ...\n', ii, numel(files), config_name);
    R = iclassic_insar(fullfile(files(ii).folder, files(ii).name), 'exported', ...
                       'NBay', n_bay, 'Seed', seed, 'StartMode', start_mode, ...
                       'OutDir', scene_out_dir, 'Plots', true, 'SaveMat', true);
    status{ii} = 'done';
    r2_values(ii) = R.r2;
    runtime_total_sec(ii) = R.runtime_total_sec;
    fprintf('  -> R^2=%.4f  depth=%.0f m  dV=%.3g m^3  (%.1f s)\n', ...
            R.r2, R.post_med(3), R.post_med(4), R.runtime_total_sec);
end

% --- [3/3] Group summary ---------------------------------------------------
summary_csv = fullfile(out_dir, sprintf('_group_summary_%s%s.csv', ...
                       strrep(strrep(group_glob, '*', ''), '_', ''), chunk_tag));
writetable(table(config_names, status, r2_values, runtime_total_sec, ...
                 'VariableNames', {'config', 'status', 'r2', 'runtime_total_sec'}), summary_csv);
fprintf('[3/3] Wrote %s\n', summary_csv);
fprintf('Done in %.1f s. Output saved to %s/\n', toc(t_group), out_dir);
