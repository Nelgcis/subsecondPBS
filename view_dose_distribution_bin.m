%% view_dose_distribution_bin.m
clear; clc; close all;

fname = './output/bg816_beam942_final_dose.bin';
thr = 0;
forceDims = [];
payloadLayout = 'auto';   % 'auto' | 'x-major' | 'z-major'

fname = normalizeTextScalar(fname);
payloadLayout = normalizeTextScalar(payloadLayout);
fname = resolveExistingDosePath(fname);

info = dir(fname);
assert(~isempty(info), 'File not found: %s', fname);
bytes = info.bytes;

fid = fopen(fname, 'rb', 'ieee-le');
assert(fid > 0, 'Cannot open %s', fname);

headerBytes = 36; % 3 int32 + 6 float32
hasHeader = false;
nx = 0; ny = 0; nz = 0;
spacing = [1 1 1];
origin = [0 0 0];

[hasHeader, nx, ny, nz, spacing, origin] = tryReadDoseHeader(fid, bytes, headerBytes);

if ~hasHeader
    [metaFound, nx, ny, nz, spacing, origin] = tryResolveCompanionMetadata(fname, headerBytes);
    if metaFound
        fprintf('No header in target file. Resolved dims/spacing/origin from companion metadata.\n');
    end

    fseek(fid, 0, 'bof');
    if nx <= 0 || ny <= 0 || nz <= 0
        if isempty(forceDims)
            if mod(bytes, 4) ~= 0
                error('Raw file size %d is not divisible by 4. Provide forceDims or check file.', bytes);
            end
            nvoxGuess = bytes / 4;
            forceDims = guessDims(nvoxGuess);
            fprintf('No header/metadata detected. Guessed dims = [%d %d %d]\n', forceDims(1), forceDims(2), forceDims(3));
        end
        nx = forceDims(1); ny = forceDims(2); nz = forceDims(3);
        spacing = [1 1 1];
        origin = [0 0 0];
    end
end

nvox = double(nx) * double(ny) * double(nz);

if hasHeader
    fseek(fid, headerBytes, 'bof');
end
rawDose = fread(fid, nvox, 'float32=>single');
fclose(fid);

assert(numel(rawDose) == nvox, 'Read %d voxels, expected %d', numel(rawDose), nvox);

layout = resolvePayloadLayout(fname, hasHeader, payloadLayout);
Dose = reshapeDosePayload(rawDose, [nx ny nz], layout);

sliceSum = squeeze(sum(sum(double(Dose), 1), 2));
sliceMax = squeeze(max(max(Dose, [], 1), [], 2));
sliceNnz = squeeze(sum(sum(Dose > thr, 1), 2));

zAxis = origin(3) + (0:nz-1) * spacing(3);

fprintf('Header: %d\n', hasHeader);
fprintf('Dims: %d x %d x %d\n', nx, ny, nz);
fprintf('Spacing: (%.6g, %.6g, %.6g)\n', spacing(1), spacing(2), spacing(3));
fprintf('Origin : (%.6g, %.6g, %.6g)\n', origin(1), origin(2), origin(3));
fprintf('Payload layout: %s\n', layout);
fprintf('Global max dose: %.6g\n', max(Dose(:)));
fprintf('Active slices (sum>thr): %d/%d\n', nnz(sliceSum > thr), nz);

figure; plot(zAxis, sliceSum, '-'); grid on;
xlabel('z'); ylabel('sum dose'); title('sum dose per z slice');

figure; plot(zAxis, sliceMax, '-'); grid on;
xlabel('z'); ylabel('max dose'); title('max dose per z slice');

figure; plot(zAxis, sliceNnz, '-'); grid on;
xlabel('z'); ylabel(sprintf('#voxels > %.2g', thr)); title('active voxels per z slice');

figure('Name', 'dose slice', 'NumberTitle', 'off');
ax = axes;
cmin = 0;
cmax = double(max(Dose(:)));
k = max(1, min(nz, round(nz / 2)));
img = imagesc(ax, Dose(:,:,k)');
axis(ax, 'image');
colormap(ax, 'parula');
colorbar(ax);
set(ax, 'CLim', [cmin cmax]);
title(ax, sprintf('z=%d (%.3f)', k-1, zAxis(k)));
xlabel(ax, 'x'); ylabel(ax, 'y');

uicontrol('Style', 'slider', 'Min', 1, 'Max', nz, 'Value', k, ...
    'Units', 'normalized', 'Position', [0.20 0.01 0.60 0.04], ...
    'SliderStep', [1/max(nz-1,1) min(10/max(nz-1,1),1)], ...
    'Callback', @(src,~) setSlice(img, ax, Dose, zAxis, round(src.Value)));

uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.82 0.01 0.16 0.04], ...
    'String', sprintf('z=%d', k-1));

function setSlice(img, ax, Dose, zAxis, kk)
    k = max(1, min(size(Dose, 3), kk));
    set(img, 'CData', Dose(:,:,k)');
    title(ax, sprintf('z=%d (%.3f)', k-1, zAxis(k)));
    drawnow;
end

function dims = guessDims(nvox)
    knownDims = [
        211 152 217
        152 211 217
        217 152 211
        223 152 217
        217 152 223
    ];
    for i = 1:size(knownDims, 1)
        if prod(double(knownDims(i,:))) == double(nvox)
            dims = knownDims(i,:);
            return;
        end
    end

    cands = [32 48 64 80 96 100 112 128 152 160 192 200 211 217 223 224 256 320 384 400 448 464 480 512];
    best = [];
    bestScore = inf;
    for x = cands
        if mod(nvox, x) ~= 0, continue; end
        rest = nvox / x;
        for y = cands
            if mod(rest, y) ~= 0, continue; end
            z = rest / y;
            if z < 4 || z > 4096, continue; end
            score = abs(x - y) + abs(z - 128);
            if score < bestScore
                bestScore = score;
                best = [x y z];
            end
        end
    end
    if isempty(best)
        best = guessDimsByFactorScan(nvox);
    end
    if isempty(best)
        error('Cannot guess dims from nvox=%d. Provide forceDims.', nvox);
    end
    dims = best;
end

function [hasHeader, nx, ny, nz, spacing, origin] = tryReadDoseHeader(fid, bytes, headerBytes)
    hasHeader = false;
    nx = 0; ny = 0; nz = 0;
    spacing = [1 1 1];
    origin = [0 0 0];

    if bytes <= headerBytes
        return;
    end

    fseek(fid, 0, 'bof');
    dimsTry = fread(fid, 3, 'int32');
    if numel(dimsTry) ~= 3 || ~all(dimsTry > 0) || ~all(dimsTry < 1e7)
        return;
    end

    spTry = fread(fid, 3, 'float32');
    orgTry = fread(fid, 3, 'float32');
    nvoxTry = double(dimsTry(1)) * double(dimsTry(2)) * double(dimsTry(3));
    if bytes ~= headerBytes + nvoxTry * 4
        return;
    end

    hasHeader = true;
    nx = dimsTry(1); ny = dimsTry(2); nz = dimsTry(3);
    spacing = double(spTry(:))';
    origin = double(orgTry(:))';
end

function [found, nx, ny, nz, spacing, origin] = tryResolveCompanionMetadata(fname, headerBytes)
    found = false;
    nx = 0; ny = 0; nz = 0;
    spacing = [1 1 1];
    origin = [0 0 0];

    [folder, base, ext] = fileparts(fname);
    if isempty(folder)
        folder = '.';
    end

    candidates = companionHeaderCandidates(folder, base, ext);
    candidates = [candidates, fallbackSiblingHeaderCandidates(fname, headerBytes)]; %#ok<AGROW>
    for i = 1:numel(candidates)
        candidate = candidates{i};
        if exist(candidate, 'file') ~= 2
            continue;
        end
        info = dir(candidate);
        fid = fopen(candidate, 'rb', 'ieee-le');
        if fid < 0
            continue;
        end
        cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
        [hasHeader, nx, ny, nz, spacing, origin] = tryReadDoseHeader(fid, info.bytes, headerBytes);
        if hasHeader
            found = true;
            return;
        end
    end

    metadataDirCandidates = companionMetadataDirs(folder, base);
    for i = 1:numel(metadataDirCandidates)
        metaDir = metadataDirCandidates{i};
        if exist(metaDir, 'dir') ~= 7
            continue;
        end
        dimsPath = fullfile(metaDir, 'doseGrid_dims.bin');
        resPath = fullfile(metaDir, 'doseGrid_resolution.bin');
        cornerPath = fullfile(metaDir, 'doseGrid_corner.bin');
        if exist(dimsPath, 'file') ~= 2 || exist(resPath, 'file') ~= 2 || exist(cornerPath, 'file') ~= 2
            continue;
        end
        dims = readBinaryVector(dimsPath, 'int32');
        res = readBinaryVector(resPath, 'float32');
        corner = readBinaryVector(cornerPath, 'float32');
        if numel(dims) >= 3 && numel(res) >= 3 && numel(corner) >= 3
            nx = double(dims(1));
            ny = double(dims(2));
            nz = double(dims(3));
            spacing = double(res(1:3));
            origin = double(corner(1:3));
            found = true;
            return;
        end
    end
end

function candidates = companionHeaderCandidates(folder, base, ext)
    candidates = {};
    if length(base) >= length('_final_dose') && strcmp(base(end-10:end), '_final_dose')
        beamPrefix = base(1:end-11);
        candidates{end+1} = fullfile(folder, [beamPrefix '_dose_grid' ext]); %#ok<AGROW>
        candidates{end+1} = fullfile(folder, ['dose_' beamPrefix ext]); %#ok<AGROW>
    elseif length(base) >= length('_dose_grid') && strcmp(base(end-9:end), '_dose_grid')
        beamPrefix = base(1:end-10);
        candidates{end+1} = fullfile(folder, ['dose_' beamPrefix ext]); %#ok<AGROW>
    end
end

function dirs = companionMetadataDirs(folder, base)
    dirs = {};
    dirs{end+1} = fullfile(folder, 'output'); %#ok<AGROW>
    dirs{end+1} = fullfile(folder, '..', 'test_data', 'output'); %#ok<AGROW>
    dirs{end+1} = fullfile(folder, '..', '..', 'test_data', 'output'); %#ok<AGROW>

    beamToken = regexp(base, '(bg\d+_beam\d+)', 'tokens', 'once');
    if ~isempty(beamToken)
        dirs{end+1} = fullfile(folder, '..', 'test_data', 'output'); %#ok<AGROW>
    end
end

function values = readBinaryVector(path, precision)
    fid = fopen(path, 'rb', 'ieee-le');
    if fid < 0
        values = [];
        return;
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    values = fread(fid, inf, [precision '=>double']);
end


function out = normalizeTextScalar(value)
    if isstring(value)
        out = char(value);
    else
        out = value;
    end
end

function resolved = resolveExistingDosePath(fname)
    resolved = fname;
    if exist(resolved, 'file') == 2
        return;
    end
    if startsWith(resolved, filesep) || ~isempty(regexp(resolved, '^[A-Za-z]:[\\/]', 'once'))
        return;
    end
    scriptPath = mfilename('fullpath');
    if isempty(scriptPath)
        return;
    end
    scriptDir = fileparts(scriptPath);
    candidate = fullfile(scriptDir, resolved);
    if exist(candidate, 'file') == 2
        resolved = candidate;
    end
end

function candidates = fallbackSiblingHeaderCandidates(fname, headerBytes)
    candidates = {};
    info = dir(fname);
    if isempty(info)
        return;
    end

    [folder, ~, ~] = fileparts(fname);
    if isempty(folder)
        folder = '.';
    end

    listing = dir(fullfile(folder, '*.bin'));
    for i = 1:numel(listing)
        if listing(i).isdir
            continue;
        end
        candidate = fullfile(listing(i).folder, listing(i).name);
        if strcmp(candidate, fname)
            continue;
        end
        if listing(i).bytes ~= info.bytes + headerBytes
            continue;
        end
        candidates{end+1} = candidate; %#ok<AGROW>
    end
end

function dims = guessDimsByFactorScan(nvox)
    dims = [];
    bestScore = inf;
    maxSide = min(1024, floor(sqrt(double(nvox))));
    for x = 16:maxSide
        if mod(nvox, x) ~= 0
            continue;
        end
        rest = nvox / x;
        maxY = min(1024, floor(sqrt(double(rest))));
        for y = 16:maxY
            if mod(rest, y) ~= 0
                continue;
            end
            z = rest / y;
            if z < 16 || z > 4096
                continue;
            end
            cand = sort([x y z]);
            score = abs(cand(1) - cand(2)) + abs(cand(3) - 192);
            if score < bestScore
                bestScore = score;
                dims = [cand(2) cand(1) cand(3)];
            end
        end
    end
end

function layout = resolvePayloadLayout(fname, hasHeader, payloadLayout)
    if ~strcmpi(payloadLayout, 'auto')
        layout = lower(payloadLayout);
        return;
    end

    [~, base, ~] = fileparts(fname);
    if length(base) >= length('_final_dose') && strcmp(base(end-10:end), '_final_dose')
        layout = 'x-major';
    elseif length(base) >= length('_dose_grid') && strcmp(base(end-9:end), '_dose_grid')
        layout = 'x-major';
    elseif length(base) >= 5 && strcmp(base(1:5), 'dose_')
        layout = 'z-major';
    elseif strcmp(base, 'dose_distribution')
        layout = 'z-major';
    elseif hasHeader
        layout = 'z-major';
    else
        layout = 'x-major';
    end
end

function Dose = reshapeDosePayload(raw, dims, layout)
    nx = dims(1); ny = dims(2); nz = dims(3);
    switch lower(layout)
        case 'z-major'
            Dose = reshape(raw, [nx, ny, nz]);
        case 'x-major'
            Dose = permute(reshape(raw, [nz, ny, nx]), [3 2 1]);
        otherwise
            error('Unknown payload layout: %s', layout);
    end
end
