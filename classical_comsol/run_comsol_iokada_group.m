% =========================================================================
% Usage:       (from the PILA repo root, normally via a jobs/comsol_classical_topo/*.pbs job)
%              SHIM_DIR=$PWD/exported_comsol/iokada_shims OUT_DIR=$PWD/classic_out_comsol_topo/varying_depth \
%              GROUP_GLOB='*_cone_desc_*' TOPO=varying_depth \
%              matlab -batch "run('classical_comsol/run_comsol_iokada_group.m')"
% Description: Classical SA + Bayesian MCMC Mogi inversion of COMSOL wk6 scenes WITH or WITHOUT
%              the Williams & Wadge (1998) varying-depth topography correction, using the
%              existing iokada_agung_residuals.m ('Model','mogi','TopoCorrection',TOPO) on the
%              HDF5 shims from classical_comsol/build_comsol_iokada_shims.py. One scene at a
%              time; scenes with an existing result are skipped (safe resume).
% Inputs:      env SHIM_DIR, OUT_DIR, GROUP_GLOB, TOPO ('none' | 'varying_depth'),
%              optional EXPORT_DIR (default <repo>/exported_comsol/single, for the bounds),
%              NBAY (default 1e4), SEED (default 1), CHUNK_INDEX / CHUNK_COUNT.
% Outputs:     OUT_DIR/<config>/mogi_los_<track>_20000101_result.mat (+ posterior png),
%              OUT_DIR/_group_summary_<glob>[_chunk].csv
% Date:        2026-10-12
%
% Scientific notes:
%   * Matched to the earlier classical baseline (run_comsol_group.m / iclassic_insar.m):
%     same points, LOS and per-pixel LOS vectors, same search box (the export's bounds),
%     NBay 1e4, seed 1, SAProfile 'june' (box-centre start; iclassic was rebuilt from this
%     sampler), likelihood sigma std(LOS). BinSize_m = 99.9 so every 100 m pixel is its own
%     bin (no averaging): floor(k*100/99.9) = k for k < 999.
%   * TOPO = 'none' is the within-code flat baseline (and a parity check vs iclassic);
%     'varying_depth' uses depth m(3) + h_i with h from the shim (COMSOL base datum, h >= 0).
%   * Depth is below the h = 0 datum (the COMSOL base) in both modes.
%   * JOINT scenes (*_joint_*, 2026-10-12): asc + desc stacked, each pixel with its own LOS
%     vector (LookVector 'per_pixel'). iokada's 'Track' only picks the constant fallback
%     vector and the file tag, so it is passed as 'asc' (results: mogi_los_asc_*). Desc pixels
%     were shifted 0.09375 m east in the shim and BinSize_m = 0.0625 keeps one pixel per bin
%     (see build_comsol_iokada_shims.py); the search box comes from exported_comsol/joint.
% =========================================================================
clear; clc; close all;

% --- Paths and settings (from the environment) -----------------------------
shim_dir   = getenv('SHIM_DIR');
out_dir    = getenv('OUT_DIR');
group_glob = getenv('GROUP_GLOB');
topo_mode  = getenv('TOPO');
if isempty(shim_dir) || isempty(out_dir) || isempty(group_glob) || isempty(topo_mode)
    error('run_comsol_iokada_group:env', 'SHIM_DIR, OUT_DIR, GROUP_GLOB and TOPO must all be set');
end
if ~any(strcmp(topo_mode, {'none', 'varying_depth'}))
    error('run_comsol_iokada_group:topo', 'TOPO must be none or varying_depth, got %s', topo_mode);
end
% run() cd's into this script's folder, so relative paths would resolve against classical_comsol/.
if ~startsWith(shim_dir, '/') || ~startsWith(out_dir, '/')
    error('run_comsol_iokada_group:relPath', 'SHIM_DIR and OUT_DIR must be absolute paths');
end
export_dir = getenv('EXPORT_DIR');
if isempty(export_dir); export_dir = fullfile(fileparts(shim_dir), 'single'); end
n_bay = str2double(getenv('NBAY'));        if isnan(n_bay); n_bay = 1e4; end
seed  = str2double(getenv('SEED'));        if isnan(seed);  seed = 1;    end
bin_size_m = 99.9;                         % one 100 m pixel per bin (see notes)

% Pin threads to the PBS allocation so runtimes are repeatable.
n_thread = str2double(getenv('NCPUS'));
if isnan(n_thread) || n_thread < 1; n_thread = 4; end
maxNumCompThreads(n_thread);

addpath('/eos-rs/INSAR_processing/denny/matlab/synthetic_deformation/Volcano_Deformation');
addpath('/eos-rs/INSAR_processing/denny/matlab/synthetic_deformation/Volcano_Deformation/modified models');

% --- [1/3] Find the scenes of this group -----------------------------------
files = dir(fullfile(shim_dir, [group_glob '.h5']));
if isempty(files)
    error('run_comsol_iokada_group:noFiles', 'no shims match %s', fullfile(shim_dir, [group_glob '.h5']));
end
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
fprintf('[1/3] %d scenes in %s matching %s%s  (TOPO=%s, NBay=%g, seed=%d, threads=%d)\n', ...
        numel(files), shim_dir, group_glob, chunk_tag, topo_mode, n_bay, seed, n_thread);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% --- [2/3] Invert each scene -----------------------------------------------
config_names = cell(numel(files), 1);
status = cell(numel(files), 1);
runtime_sec = nan(numel(files), 1);
t_group = tic;
for ii = 1:numel(files)
    [~, config_name] = fileparts(files(ii).name);
    config_names{ii} = config_name;
    tokens = regexp(config_name, '_(asc|desc|joint)_', 'tokens', 'once');
    track = tokens{1};
    if strcmp(track, 'joint')
        iokada_track = 'asc';          % file tag only; per-pixel LOS vectors drive the physics
        scene_bin_size_m = 0.0625;     % must match JOINT_BIN_SIZE_M in the shim builder
        scene_export_dir = fullfile(fileparts(export_dir), 'joint');
        expected_bins = 2 * 40401;
    else
        iokada_track = track;
        scene_bin_size_m = bin_size_m;
        scene_export_dir = export_dir;
        expected_bins = 40401;
    end
    scene_out_dir = fullfile(out_dir, config_name);
    result_mat = fullfile(scene_out_dir, sprintf('mogi_los_%s_20000101_result.mat', iokada_track));
    if exist(result_mat, 'file')
        % Never overwrite a finished result (iokada would back it up, but skip is cleaner).
        fprintf('[2/3] (%d/%d) %s: result exists, skipped\n', ii, numel(files), config_name);
        status{ii} = 'skipped';
        continue
    end
    % Same search box as the iclassic baseline (from the matching export).
    export_mat = fullfile(scene_export_dir, [config_name '.mat']);
    if ~exist(export_mat, 'file')
        error('run_comsol_iokada_group:noExport', 'missing export for bounds: %s', export_mat);
    end
    E = load(export_mat, 'bounds');
    fprintf('[2/3] (%d/%d) %s (track %s) ...\n', ii, numel(files), config_name, track);
    t_scene = tic;
    iokada_agung_residuals(fullfile(files(ii).folder, files(ii).name), '20000101', 'los', ...
        'Model', 'mogi', 'Track', iokada_track, 'OutDir', scene_out_dir, ...
        'BinSize_m', scene_bin_size_m, 'NBay', n_bay, 'Seed', seed, 'Bounds', double(E.bounds), ...
        'TopoCorrection', topo_mode, 'SAProfile', 'june', 'LookVector', 'per_pixel');
    runtime_sec(ii) = toc(t_scene);
    R = load(result_mat, 'post_med', 'x_ds');
    if numel(R.x_ds) ~= expected_bins
        % Binning must not merge pixels, or the data differ from the iclassic baseline.
        error('run_comsol_iokada_group:binning', '%s: %d bins, expected %d', config_name, numel(R.x_ds), expected_bins);
    end
    status{ii} = 'done';
    fprintf('  -> x=%.0f m  y=%.0f m  depth=%.0f m  dV=%.3g m^3  (%.1f s)\n', ...
            R.post_med(1), R.post_med(2), R.post_med(3), R.post_med(4), runtime_sec(ii));
end

% --- [3/3] Group summary ---------------------------------------------------
summary_csv = fullfile(out_dir, sprintf('_group_summary_%s%s.csv', ...
                       strrep(strrep(group_glob, '*', ''), '_', ''), chunk_tag));
writetable(table(config_names, status, runtime_sec, ...
                 'VariableNames', {'config', 'status', 'runtime_sec'}), summary_csv);
fprintf('[3/3] Wrote %s\n', summary_csv);
fprintf('Done in %.1f s. Output saved to %s/\n', toc(t_group), out_dir);
