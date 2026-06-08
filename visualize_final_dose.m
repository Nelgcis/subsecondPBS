% visualize_final_dose.m
% Visualize 3D dose grid slices from raw float32 binary (no header).

% ---- User settings ----
filePath = 'output/bg748_beam873_final_dose.bin';
% Dimensions must match the volume used during the run:
% For the CSV path in wrapper_integration_test.cu, this is 512x512x256.
% If you change im/dose dims, update these values.
nx = 512;
ny = 512;
nz = 256;

% Slice indices (1-based). Leave empty to use center slices.
ix = [];
iy = [];
iz = [];

% ---- Load ----
fid = fopen(filePath, 'rb');
if fid < 0
    error('Cannot open file: %s', filePath);
end
raw = fread(fid, nx * ny * nz, 'float32=>single');
fclose(fid);

if numel(raw) ~= nx * ny * nz
    error('Unexpected file size. Read %d floats, expected %d.', numel(raw), nx * ny * nz);
end

% Reshape: x-fastest, then y, then z.
dose = reshape(raw, [nx, ny, nz]);

% Default slice positions
if isempty(ix), ix = round(nx / 2); end
if isempty(iy), iy = round(ny / 2); end
if isempty(iz), iz = round(nz / 2); end

% ---- Visualize ----
figure('Color', 'w', 'Name', 'Dose Slices');

subplot(1,3,1);
imagesc(squeeze(dose(:,:,iz))');
axis image; colormap turbo; colorbar;
title(sprintf('Axial (z = %d)', iz));
xlabel('x'); ylabel('y');

subplot(1,3,2);
imagesc(squeeze(dose(:,iy,:))');
axis image; colormap turbo; colorbar;
title(sprintf('Sagittal (y = %d)', iy));
xlabel('x'); ylabel('z');

subplot(1,3,3);
imagesc(squeeze(dose(ix,:,:))');
axis image; colormap turbo; colorbar;
title(sprintf('Coronal (x = %d)', ix));
xlabel('y'); ylabel('z');

sgtitle('3D Dose Grid Slices');
