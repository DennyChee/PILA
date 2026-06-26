% Usage:       matlab -batch "regen_noise_cube"
% Description: Regenerate the Marapi noise cube with a CORRECTED orbital ramp.
%              The committed noise_cube.mat used an old P_orb with sinusoidal
%              stripes (too many cycles) for the 'orbit' field. This script keeps
%              the physically-derived trop (ERA5/MatAPS) + turbulent atm + white
%              noise from the existing cube, regenerates 'orbit' as a clean
%              degree-1 random plane (NO stripes) via the canonical orbital_error.m,
%              recombines, and saves noise_cube_v2.mat. Orientation is left in the
%              native MATLAB grid order; the Python loader applies the north-up
%              flipud fix (validated against the Marapi DEM).
% Date:        2026-06-22

clear; clc;

noise_dir = '/eos-rs/INSAR_processing/denny/matlab/synthetic_deformation/synthetic_noise';
addpath(noise_dir);                              % for orbital_error.m
src  = fullfile(noise_dir, 'noise_cube.mat');
dst  = fullfile(noise_dir, 'noise_cube_v2.mat');

fprintf('[1/4] Loading existing cube: %s\n', src);
S = load(src);
nc = S.noise_cube;
trop  = nc.trop;          % [Ny,Nx,Nt] stratified APS (ERA5/MatAPS) - KEEP
atm   = nc.atm;           % turbulent APS - KEEP
white = nc.noise;         % white - KEEP
[Ny, Nx, Nt] = size(trop);
fprintf('  grid %d x %d, %d epochs\n', Ny, Nx, Nt);

fprintf('[2/4] Regenerating orbital ramp (degree-1 plane, NO stripes)\n');
% Same 40 km / 128 px grid the cube was built on.
[x, y] = meshgrid(linspace(0, 40000, Nx), linspace(0, 40000, Ny));
t = 1:Nt;
% Clean linear ramp: random plane orientation per epoch, 8 mm RMS, stripes OFF.
P_orb = struct( ...
    'degree',        1, ...      % first-order plane (a single gradient = 1 'cycle')
    'sigma',         0.008, ...  % 8 mm RMS ramp
    'randomize_dir', true, ...   % independent ramp each epoch
    'keep_angle',    false, ...
    'stripe_amp',    0.0);       % NO stripes (this was the bug)
Orb = orbital_error(x, y, t, P_orb, 42);         % [Ny,Nx,Nt], metres LOS
fprintf('  orbit RMS (median over epochs) = %.2f mm\n', ...
        1000 * median(squeeze(sqrt(mean(reshape(Orb, [], Nt).^2, 1)))));

fprintf('[3/4] Recombining: combined = trop + atm + orbit + white\n');
combined = trop + atm + Orb + white;

fprintf('[4/4] Saving %s\n', dst);
noise_cube = struct();
noise_cube.combined = combined;
noise_cube.trop  = trop;
noise_cube.atm   = atm;
noise_cube.orbit = Orb;        % regenerated, stripe-free
noise_cube.noise = white;
noise_cube.provenance = ['orbit regenerated stripe-free via orbital_error.m; ' ...
                         'trop/atm/white kept from noise_cube.mat'];
save(dst, 'noise_cube', '-v7');
fprintf('Done. Wrote %s\n', dst);
