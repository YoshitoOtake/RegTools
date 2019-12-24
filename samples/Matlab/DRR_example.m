volumeFileName = 'K:\Projects\StatisticalDeformationModel\dataset\Head_1.0mm\PCA_mode\MeanVolume_head_1.0mm_no_left_out.mhd';
[volume, header] = mhdread(volumeFileName);

addpath('C:\Users\yoshi\Documents\CUDA_Programs\RegTools_build_VS2008\Matlab');

% initialize RegTools
regTools = RegTools(-1,[],'log_file.txt');
regTools.SetVolumeInfo( struct('VolumeDim', size(volume), 'VoxelSize', header.ElementSpacing ) );

% setup projection geometry
SDD = 1000; SAD = 700;
DRR_size = [768 768]; DRR_Pixel_size = [0.38 0.38];
extrinsic_AP = RegTools.matTranslation([0 0 -SAD])*RegTools.matRotationY(-90)*RegTools.matRotationX(90);
extrinsic_LAT = RegTools.matTranslation([0 0 -SAD])*RegTools.matRotationY(0)*RegTools.matRotationX(90);
intrinsic = [-SDD/DRR_Pixel_size(1) 0 DRR_size(1)/2; 0 -SDD/DRR_Pixel_size(2) DRR_size(2)/2; 0 0 1];
ProjectionMatrices = cat(3, [intrinsic [0;0;0]] * extrinsic_AP, [intrinsic [0;0;0]] * extrinsic_LAT);
regTools.GenerateGeometry_3x4ProjectionMatrix( ProjectionMatrices, DRR_Pixel_size(1), DRR_size);

% create DRR
DRRs = regTools.ForwardProject( max(volume, 0), eye(4), [], size(ProjectionMatrices,3) );

% cleanup
clear regTools

%%
figure('Name', 'DRR example', 'Position',[100 150 1600 800], 'PaperPositionMode', 'auto', 'InvertHardcopy', 'off', 'Color', 'k' );
colormap(gray(256));
subplot(1,2,1);
imagesc(DRRs(:,:,1)'); axis image;
set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
subplot(1,2,2);
imagesc(DRRs(:,:,2)'); axis image;
set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w');
