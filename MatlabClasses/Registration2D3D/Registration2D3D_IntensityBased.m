classdef Registration2D3D_IntensityBased < Registration2D3DInterface & handle
    %REGISTRATION2D3D_INTENSITYBASED - Intensity-based (rigid) 2D/3D registration
    %algorithm.
    
    properties
        % Objective function name
        ObjectiveFunctionName = 'Registration2D3D_IntensityBased.ComputeObjectiveFunction';
        
        % Number of multi resolution levels
        NumMultiLevel = 1;
        
        % Number of iterations for each resolution level
        NumIterations = [];
        
        % the value of the cost function at the solution
        fmin = 0;
        
        % parameters for CMA-ES optimizer
        InSigmas = [];
        MaxEval = [];
        PopSize = (4 + floor(3*log(6))); % population size
        TolX = [];
        TolFun = [];
        MultiStart = [];
        IntermediateTransform = [];
        UBounds = Inf;
        LBounds = -Inf;
        SearchRange = [];
        NumOptimizationLayers = 1;  % if >=2, repeat entire optimization multiple times using local minima found in the previous loop as initialization
        
        % setting of down-sampling and 'sub-region projection'
        ROI_box_ratios = [];
        DownsampleRatio_Projection = [];
        DownsampleRatio_Volume = [];
        InitialSearch_DownsampleRatio = []; % [projection volume]
        MaxParallel = [];
        
        % setting for initialization search
        NumInitialSearchCandidate = 3;
        InitialSearch_PeakDetectThreshold = 0.95;
        
        % default setting for GI parameter
        GI_Sigma = 1.0;
        GI_threshold = -3.402823466e+38;     % definition of -FLT_MAX from <float.h>
        
        % default setting for SSIM
        SSIM_DynamicRange = 1.0;
        
        % default setting for MI
        MI_NumberOfBins = 64;
        
        % type of optimizer (currently only 'CMA-ES' and 'fminsearch' are
        % supported
        Optimizer = 'CMA-ES';
        
        % initial simplex size for fminsearch
        fminsearch_initial_size = [];
        
        % mask weight for similarity measure computation
        % single precision array of the same size as fixed images
        SimilarityMeasureMask = [];
        
        % seed for random sequence used in CMA-ES 
        RandSeed = -1;
        
        % static function for cropping (takes current guess, input images, and RegTools object
        % and outputs cropped image)
        CroppingFunctionName = 'Registration2D3D_IntensityBased.CropProjections';        
        
        % static function for setting optimization parameters
        OptimizationSettingFunctionName = 'Registration2D3D_IntensityBased.SetOptimizationSettingFunction';
        
        % static function for displaying initial guess
        % (Note: the function name should not be "ShowDRR" because
        % it already exists.)
        ShowDRRFunctionName = '';
        
        % static function for computing projection matrices
        GetProjectionMatricesFunctionName = 'Registration2D3D_IntensityBased.GetProjectionMatrices_SingleImage2D3D';
        
        % for multi start
        SuccessfulStart = 0;
        
        % values returned from CMA-ES for confidence metric
        CMAES_sigma = [];
        CMAES_B = [];
        CMAES_diagD = [];
        CMAES_C = [];
        MultistartCMAES_myu = -1;    % selection ratio in multistart CMA-ES
        MultistartRanges = [];
        Median_m_Best = 0;
        AllMultistartSolutions = [];
        AllMultistartFmins = [];
        
        % if debug mode or not
        Debug = false;
        
        % serial function evaluation mode in CMA-ES
        % very slow, only for debugging
        SerialCMAES = false;
        
        % values for RayCasting projector
        RayCastingThreshold = -500;
        RayCastingDistanceFalloffCoefficient = -500;
        
        % count pixels that is not covered by any voxels
        CountNonIntersectedPixel = false;
        
        ProjectorMode = RegTools.ProjectorMode_LinearInterpolation;
    end
    
    properties (Access = private)
        all_images;  % downsampled images (cell array)
        all_volumes; % downsampled volumes (cell array)
        all_similarity_masks; % downsampled mask weight for similarity measure computation
        
        initial_search_image;
        initial_search_mask;
        initial_search_volume;
        
        % object of RegTools class
        regTools = -1;
        keepRegToolsObject = 0;   % flag indicating if RegTools object is constructed externally or not
    end
    
    %%%%%%%%%%%%%%%%%%
    % Public methods %
    %%%%%%%%%%%%%%%%%%
    methods
        function obj = Registration2D3D_IntensityBased()
            %REGISTRATION2D3D_INTENSITYBASED - Constructor for class
            obj.Param = Registration2D3D_IntensityBased_ObjectiveFunctionParameters();
        end % Registration2D3D_IntensityBased
        function SetInitialGuess(obj, guess)
            if nargin < 2 || isempty(guess)
                guessGUI = Registration2D3DInitialGuess(...
                    obj.InputVolume, ...
                    obj.VoxelSize, ...
                    obj.Images, ...
                    obj.ProjectionMatrices );

                guessGUI.PixelSize = obj.PixelSize;

                % this is the center of the femoral head in mm space
                guessGUI.VolumeCoordinateOffset = obj.Param.VolumeCoordinateOffset;

                guessGUI.InitializeRegTools();
                
                % now make it wait until the figure is closed
                guessGUI.ReturnOnClose = true;                
                uiwait(guessGUI.hg.hFig);
                
                guess = ...
                    guessGUI.VolumeCoordinateOffset * guessGUI.CurrentGuess / guessGUI.VolumeCoordinateOffset;
                
                delete(guessGUI);
            end
            obj.InitialGuess = guess;
            obj.Param.InitialGuess = guess;
        end % SetInitialGuess
        
        function SetProjectionMatrices(obj, pms)
            obj.ProjectionMatrices = pms;
            obj.Param.OriginalProjectionMatrices = pms;
        end % SetProjectionMatrices
        
        function pms = GetProjectionMatrices(obj)
            pms = obj.ProjectionMatrices;
        end
        
        function images = GetImages(obj)
            images = obj.Images;
        end
        
        function SetImages(obj, images)
            if(~exist('images','var') || isempty(images))
                obj.Images = [];
            else
                obj.Images = images;
                obj.Param.NumImages = size(obj.Images, 3);
                obj.Param.ImageDim = [size(obj.Images,1) size(obj.Images,2)];
            end
        end % SetImages
        
        function images = GetDownsampledImages(obj, DS_level)
            if(~exist('DS_level','var')), DS_level = length(obj.all_images); end % return the last level
            images = obj.all_images{DS_level};
        end
        
        function images = GetDownsampledMasks(obj, DS_level)
            if(~exist('DS_level','var')), DS_level = length(obj.all_similarity_masks); end % return the last level
            images = obj.all_similarity_masks{DS_level};
        end
        
        function volume = GetDownsampledVolume(obj, DS_level)
            if(~exist('DS_level','var')), DS_level = length(obj.all_volumes); end % return the last level
            volume = obj.all_volumes{DS_level};
        end
        
        function SetSimilarityMeasureMask(obj, mask)
            if size(mask,3) == 1
                mask = repmat(mask, [1 1 size(obj.Images,3)]);
            end
            obj.SimilarityMeasureMask = mask;
        end % SetSimilarityMeasureMask
        
        function SetPixelSize(obj, pixel_size)
            obj.PixelSize = pixel_size;
            obj.Param.PixelSize = pixel_size;
        end % SetPixelSize
        
        function SetVoxelSize(obj, voxel_size)
            obj.VoxelSize = voxel_size;
            obj.Param.VoxelSize = voxel_size;
        end % SetVoxelSize
        
        function InitializeRegTools(obj, varargin)
            % if RegTools object is provided externally, we do not delete
            % the object in destructor (set keepRegToolsObject as 1)
            if(obj.regTools == -1)
                if(nargin>1 && isa(varargin{1}, 'RegTools'))
                    obj.regTools = varargin{1};
                    obj.keepRegToolsObject = 1;   % not delete RegTools object in destructor
                else
                    obj.regTools = RegTools(varargin{:});
                    obj.keepRegToolsObject = 0;   % delete RegTools object in destructor
                end
            end
        end
        
        function Initialize(obj, varargin)
            % if RegTools is not inintialized yet, we construct it
            obj.InitializeRegTools(varargin{:})
            
            % prepare (downsample) fixed images and floating volume for all level
            [obj.all_images, obj.all_similarity_masks] = obj.DownsampleProjections();
            obj.all_volumes = obj.DownsampleVolumes();
            
            if(~isempty(obj.InitialSearch_DownsampleRatio))
                obj.initial_search_image = obj.regTools.Downsample2DProjections( obj.InitialSearch_DownsampleRatio(1), obj.Images );
                obj.initial_search_mask = obj.regTools.Downsample2DProjections( obj.InitialSearch_DownsampleRatio(1), obj.SimilarityMeasureMask, RegTools.InterpolatorType_NearestNeighbor );
                obj.initial_search_volume = obj.DownsampleVolume( obj.InitialSearch_DownsampleRatio(2) );
            end
        end
        
        function HU2Myu_InputVolume(obj, myu_water)
            if(iscell(obj.InputVolume))
                for i=1:length(obj.InputVolume)
                    obj.InputVolume{i} = RegTools.HU2Myu(obj.InputVolume{i}, myu_water);
                end
            else
                obj.InputVolume = RegTools.HU2Myu(obj.InputVolume, myu_water);
            end
        end
        
        function LogCorrection_InputImage(obj, I0)
            obj.Images = Registration2D3D_IntensityBased.LogCorrection(obj.Images, I0);
        end
        
        function img = GetInitialSearchImages(obj)
            img = obj.initial_search_image;
        end
        
        function UninitializeRegTools(obj)
            if(~obj.keepRegToolsObject)
                obj.regTools = -1;
            end
        end
        
        function Uninitialize(obj)
            obj.UninitializeRegTools();
        end
        
        function delete(obj)
            obj.Uninitialize();
        end

        function val = isInitialized(obj)
            val = (obj.regTools ~= -1);
        end
        
        function all_volumes = DownsampleVolumes(obj)
            all_volumes = cell(obj.NumMultiLevel, 1);
            if(iscell(obj.InputVolume))
                numVolumes = length(obj.InputVolume);
                inputVolume = obj.InputVolume;
            else
                numVolumes = 1;
                inputVolume = cell(1,1);
                inputVolume{1} = obj.InputVolume;
            end
            volumeInfo = obj.regTools.GetVolumeInfo();
            for j=1:numVolumes
                volumePlan = obj.regTools.CreateInterpolatorPlan( inputVolume{j}, obj.VoxelSize(j,:) );
                for i=1:min(obj.NumMultiLevel, length(obj.DownsampleRatio_Volume))
                    if(iscell(obj.DownsampleRatio_Volume(i))), DS_u = obj.DownsampleRatio_Volume{i}; else DS_u = ones(1,3) * obj.DownsampleRatio_Volume(i); end
                    if(all(DS_u==1))
                        all_volumes{i,j} = inputVolume{j};
                    else
                        obj.regTools.SetVolumeInfo( struct('VolumeDim', ceil(size(inputVolume{j})./DS_u), 'VoxelSize', obj.VoxelSize(j,:).*DS_u) );
                        all_volumes{i,j} = obj.regTools.Interpolation( volumePlan, [0 0 0 0 0 0], RegTools.InterpolatorType_Bicubic, 0, -0.5 ); % bi-cubic interpolation
                        all_volumes{i,j} = max(0, all_volumes{i,j});    % negative value could happen due to interpolation
                    end
%                     obj.VoxelSize(i,:) = obj.VoxelSize(1,:).*DS_u;
                end
                obj.regTools.DeleteInterpolatorPlan( volumePlan );
            end
            obj.regTools.SetVolumeInfo(volumeInfo);
        end
        
        function [all_projections all_similarity_masks] = DownsampleProjections(obj)
            all_projections = cell(obj.NumMultiLevel, 1);
            all_similarity_masks = cell(obj.NumMultiLevel, 1);
            for i=1:min(obj.NumMultiLevel, length(obj.DownsampleRatio_Projection))
                all_projections{i} = obj.regTools.Downsample2DProjections(obj.DownsampleRatio_Projection(i), obj.Images);
                if(~isempty(obj.SimilarityMeasureMask))
                    all_similarity_masks{i} = obj.regTools.Downsample2DProjections(obj.DownsampleRatio_Projection(i), obj.SimilarityMeasureMask, RegTools.InterpolatorType_NearestNeighbor);
                end
            end
        end
        
        function UpdateProjections(obj)
            [obj.all_images, obj.all_similarity_masks] = obj.DownsampleProjections();
            if(~isempty(obj.InitialSearch_DownsampleRatio))
                obj.initial_search_image = obj.regTools.Downsample2DProjections( obj.InitialSearch_DownsampleRatio(1), obj.Images );
                if(~isempty(obj.SimilarityMeasureMask))
                    obj.initial_search_mask = obj.regTools.Downsample2DProjections( obj.InitialSearch_DownsampleRatio(1), obj.SimilarityMeasureMask, RegTools.InterpolatorType_NearestNeighbor );
                end
            end
        end
        
        function UpdateVolumes(obj)
            obj.all_volumes = obj.DownsampleVolumes();
        end
        
        function u_out = DownsampleVolume(obj, DS_u)
            % DS_u: down-sample ratio for volume
            if(isscalar(DS_u)), DS_u = ones(1,3)*DS_u; end
            if(all(DS_u == 1.0)), u_out = obj.InputVolume; return; end
            if(iscell(obj.InputVolume))
                u_out = cell(length(obj.InputVolume),1);
                for i=1:length(obj.InputVolume)
                    % downsampling of the volume
                    volumePlan = obj.regTools.CreateInterpolatorPlan( obj.InputVolume{i}, obj.VoxelSize(i,:) );
                    obj.regTools.SetVolumeInfo( struct('VolumeDim', ceil(size( obj.InputVolume{i} )./DS_u), 'VoxelSize', obj.VoxelSize(i,:).*DS_u) );
                    u_out{i} = obj.regTools.Interpolation( volumePlan, [0 0 0 0 0 0], RegTools.InterpolatorType_Bicubic, 0, -0.5 ); % bi-cubic interpolation
                    obj.regTools.DeleteInterpolatorPlan( volumePlan );
                end
            else
                % downsampling of the volume
                volumePlan = obj.regTools.CreateInterpolatorPlan( obj.InputVolume, obj.VoxelSize );
                obj.regTools.SetVolumeInfo( struct('VolumeDim', ceil(size( obj.InputVolume )./DS_u), 'VoxelSize', obj.VoxelSize.*DS_u) );
                u_out = obj.regTools.Interpolation( volumePlan, [0 0 0 0 0 0], RegTools.InterpolatorType_Bicubic, 0, -0.5 ); % bi-cubic interpolation
                obj.regTools.DeleteInterpolatorPlan( volumePlan );
            end
        end
        
        function similarity_measure_plan_id = CreateSimilarityMeasureComputationPlan(obj, SimilarityMeasureType, image, mask, num_image_sets)
            % prepare fixed images
            % compute images that are necessary for specified similarity
            % measure (i.e. gradient images for GI, normalized images for MI
            % and NMI, measurement image for LogLikelihood)
            switch(SimilarityMeasureType)
                case {RegTools.SimilarityMeasureType_GI, RegTools.SimilarityMeasureType_GI_SINGLE, ...
                        RegTools.SimilarityMeasureType_GI_SINGLE_ALWAYS_FLOATING_NORM, RegTools.SimilarityMeasureType_SSIM}
                    similarity_measure_plan_id = obj.regTools.CreateSimilarityMeasureComputationPlan( image, obj.GI_Sigma, mask, ...
                        num_image_sets, obj.GI_threshold, [], [], [], [], obj.SSIM_DynamicRange, SimilarityMeasureType ); % don't use normalization
                case {RegTools.SimilarityMeasureType_MI, RegTools.SimilarityMeasureType_NMI }
                    similarity_measure_plan_id = obj.regTools.CreateSimilarityMeasureComputationPlan( image, -1, mask, ...
                        num_image_sets, [], 0, 0, 0, 0, [], SimilarityMeasureType ); % don't use kernel convolution
                case {RegTools.SimilarityMeasureType_NCC, RegTools.SimilarityMeasureType_GC }
                    similarity_measure_plan_id = obj.regTools.CreateSimilarityMeasureComputationPlan( image, obj.GI_Sigma, mask, ...
                        num_image_sets, [], 0, 0, 0, 0, [], SimilarityMeasureType ); % use both normalization and kernel convolution
            end
        end
        
        function SetOptimizationSetting(obj, varargin)
            if(~isempty(obj.OptimizationSettingFunctionName))
                feval(obj.OptimizationSettingFunctionName, obj, varargin{:});
            end
        end
        
        function SetProjectionMatricesEx(obj, param)
            if(~isempty(obj.GetProjectionMatricesFunctionName))
                pm = feval(obj.GetProjectionMatricesFunctionName, obj.Param.ImageDim, obj.PixelSize, param);
                obj.SetProjectionMatrices(pm);
            end
        end
        
        function ShowInitialGuess(obj)
            obj.ShowDRR(obj.Param.InitialGuess);
        end
        
        function ShowRegistrationTransform(obj)
            obj.ShowDRR(obj.RegistrationTransform);
        end
        
        function ShowDRR(obj, transform)
            if(~isempty(obj.ShowDRRFunctionName))
                obj.InitializeRegTools();
                
                % initialize RegTools for DRR rendering
                geomID = obj.regTools.GenerateGeometry_3x4ProjectionMatrix( obj.ProjectionMatrices, obj.PixelSize, obj.Param.ImageDim );
                if(~isempty(obj.Images))
                    obj.Param.similarity_measure_plan_id = obj.CreateSimilarityMeasureComputationPlan( obj.Param.SimilarityMeasureType, obj.Images, [], 1);
                else
                    obj.Param.similarity_measure_plan_id = -1;
                end
                obj.SetStepSize(mean(obj.VoxelSize(:))); % DS_u = 1
                if(iscell(obj.InputVolume))
                    for j=1:size(obj.InputVolume,2)
                        obj.Param.forwardProjection_plan_id(j) = obj.regTools.CreateForwardProjectionPlan( obj.InputVolume{1,j}, obj.VoxelSize(j,:) );
                    end
                    feval(obj.ShowDRRFunctionName, obj.regTools, obj, transform, obj.Param);
                    for j=1:size(obj.all_volumes,2)
                        obj.regTools.DeleteForwardProjectionPlan( obj.Param.forwardProjection_plan_id(j) );
                    end
                else
                    obj.Param.forwardProjection_plan_id(1) = obj.regTools.CreateForwardProjectionPlan( obj.InputVolume, obj.VoxelSize(1,:) );
                    feval(obj.ShowDRRFunctionName, obj.regTools, obj, transform, obj.Param);
                    obj.regTools.DeleteForwardProjectionPlan( obj.Param.forwardProjection_plan_id(1) );
                end
                % clean up
                if(~isempty(obj.Images)), obj.regTools.DeleteSimilarityMeasureComputationPlan( obj.Param.similarity_measure_plan_id ); end
                obj.regTools.DeleteProjectionParametersArray( geomID );
            end
        end
        
        function SetStepSize(obj, DS_u)
            % obj.Param.StepSize>0:  LinearInterpolation projector with specified step size
            % obj.Param.StepSize==0: Error
            % obj.Param.StepSize<0:  LinearInterpolation projector with automatically defined step size (voxel size in X-direction)
            if(obj.Param.StepSize==0)
                disp('Error at SetStepSize(), step size should be larger than zero');
            else
                if(obj.Param.StepSize>0)
                    obj.regTools.SetStepSize(obj.Param.StepSize);
                else
                    obj.regTools.SetStepSize(mean(mean(obj.VoxelSize*diag(DS_u)))*2);   % Auto mode (mean of voxel length times 2)
                end
            end
        end
        
        function Run(obj)
%             disp('start intensity-based 2D/3D registration');
            % run multi-level (hierarchical) 3D/2D registration
            
            if(obj.Param.RecordLog)
                % clear RegistrationLogs
                obj.Param.RegistrationLogs = [];
                obj.Param.RegistrationLogs_LevelEnd = zeros(obj.NumMultiLevel,1);
            end
            obj.Param.current_iteration = 0;
            obj.Param.record_file_id = 0;
            obj.Param.TimerID = tic;
            % multi start position setting
            if(~isempty(obj.MultiStart))
                obj.Param.num_multi_start = size(obj.MultiStart,1);
                start_guesses = obj.MultiStart;
            else
                obj.Param.num_multi_start = 1;
                start_guesses = zeros(1,length(obj.InitialGuess));
            end
            % run multi level 3D/2D registration
            obj.Param.NextProgressShow = 0;
            obj.Param.NextProgressShowTime = 0;
            obj.IntermediateTransform = zeros(obj.NumMultiLevel,length(obj.Param.InitialGuess));
            obj.RegistrationTime = zeros(1, obj.NumMultiLevel);
            obj.NumIterations = zeros(1, obj.NumMultiLevel);
            obj.Param.num_multi_level = obj.NumMultiLevel;
            obj.regTools.SetCountNonIntersectedPixel(obj.CountNonIntersectedPixel);
            for i=1:obj.NumMultiLevel    % loop for each level
                obj.Param.current_level = i;
                if(iscell(obj.DownsampleRatio_Volume(i))), DS_u = obj.DownsampleRatio_Volume{i}; else DS_u = ones(1,3)*obj.DownsampleRatio_Volume(i); end
                [nu, nv, nb] = size(obj.all_images{i});    % dimensions of downsampled images at this level
                [nx, ny, nz] = size(obj.all_volumes{i});   % dimensions of downsampled volume at this level
                if(length(obj.PopSize)<i), popSize = obj.PopSize(end); else popSize = obj.PopSize(i); end  % population size for CMA-ES optimizer
                true_downsample_ratio = [size(obj.Images,1) size(obj.Images,2)]./[nu nv];  % when the downsample ratio is not a factor of image size, this can be non-integer
                obj.Param.CurrentDownsampleRatio_Projection = true_downsample_ratio;
                obj.SetStepSize(DS_u);
                obj.regTools.SetRayCastingThreshold(obj.RayCastingThreshold);
                obj.regTools.SetRayCastingDistanceFalloffCoefficient(obj.RayCastingDistanceFalloffCoefficient);
                
                for j=1:size(obj.all_volumes, 2)
                    obj.Param.forwardProjection_plan_id(j) = obj.regTools.CreateForwardProjectionPlan( obj.all_volumes{i,j}, obj.VoxelSize(j,:).*DS_u );
                end
                obj.regTools.SetVolumeInfo( struct('VolumeDim', size(obj.all_volumes{i}), 'VoxelSize', obj.VoxelSize.*repmat(DS_u, [size(obj.VoxelSize,1), 1])) );
                obj.regTools.SetProjectorMode(obj.ProjectorMode);

%                 obj.regTools.GPUmemCheck();
                if(i==1)
                    num_current_multi_start = obj.Param.num_multi_start;
                else
                    num_current_multi_start = 1; 
                end
                fmins = zeros(num_current_multi_start,1);
                cmaes_sigmas = zeros(num_current_multi_start);
                dimension = length(obj.Param.InitialGuess);
                cmaes_Bs = zeros(dimension, dimension, num_current_multi_start);
                cmaes_diagDs = zeros(dimension, num_current_multi_start);
                cmaes_C = zeros(dimension, dimension, num_current_multi_start);
                searched_multi_start = zeros(size(start_guesses,2), num_current_multi_start);
                
                if(strcmp(obj.Optimizer, 'CMA-ES-PARALLEL'))
                    parallelEval = popSize * num_current_multi_start;
                else
                    parallelEval = popSize;
                end
                
                if(length(obj.MaxParallel)<i || obj.MaxParallel(i)<0)
                    obj.Param.MaxParallel = parallelEval; 
                else
                    obj.Param.MaxParallel = min(parallelEval, obj.MaxParallel(i)); 
                end
                    
                % geometry setting (based on the currentGuess)
                geomID = obj.regTools.GenerateGeometry_3x4ProjectionMatrix( obj.ProjectionMatrices, obj.PixelSize.*true_downsample_ratio, [nu nv], true_downsample_ratio );
                if(length(obj.ROI_box_ratios)>=i && obj.ROI_box_ratios(i)>0)
                    % crop projection images using current guess
                    current_start_poses4x4 = RegTools.convertTransRotTo4x4_multi(start_guesses(1,1:6)', RegTools.convertTransRotTo4x4(obj.Param.InitialGuess(1:6)'), 'Post');
                    current_start_poses = [RegTools.convert4x4ToTransRot(current_start_poses4x4) start_guesses(1,7:end)+obj.Param.InitialGuess(7:end)'];
                    [obj.all_images{i}, obj.Param.ROI_left_bottoms, obj.Param.ROI_square_size] = ...
                        feval(obj.CroppingFunctionName, obj.all_images{i}, current_start_poses, [nx ny nz].*obj.VoxelSize(i,:).*DS_u, obj.ROI_box_ratios(i), obj.regTools, obj.Param);
                    obj.Param.ROI_left_bottoms = double(obj.Param.ROI_left_bottoms) * diag(true_downsample_ratio);
                    obj.Param.ROI_square_size = double(obj.Param.ROI_square_size') * diag(true_downsample_ratio);
                end

                obj.regTools.ReplicateProjections(obj.Param.MaxParallel);
                obj.regTools.SetTransferBlockSize(obj.regTools.GetNumberOfProjections());   % error if this is grater than "MAX_TRANSFER_BLOCK_SIZE"

                obj.Param.similarity_measure_plan_id = obj.CreateSimilarityMeasureComputationPlan( ...
                    obj.Param.SimilarityMeasureType, obj.all_images{i}, obj.all_similarity_masks{i}, obj.Param.MaxParallel);
                if(obj.Param.similarity_measure_plan_id<0), return; end     % error

                freeMem_byte = obj.GPUmemCheck('Optimization start');
                fprintf('Optimization start: %.2f MB available on the GPU\n', freeMem_byte/1024/1024);
                if(obj.RandSeed < 0)
                    s = RandStream('mt19937ar','Seed',sum(100*clock));
                    RandStream.setGlobalStream(s);
                    obj.RandSeed = randi(32767);
                end
                
                % Generate fixed volumes DRR
                
                obj.Param.FixedVolumesDRR = [];
                fixedVolumesIndices = obj.Param.FixedVolumesIndices;
                numFixedVolumes = length(fixedVolumesIndices);
                for j = 1:numFixedVolumes
                    fixedIndex = fixedVolumesIndices(j);
                    fixedTransform = RegTools.convertTransRotTo4x4(obj.Param.FixedVolumesTransforms(:, j));
                    if j == 1
                        obj.Param.FixedVolumesDRR = obj.regTools.ForwardProject(obj.Param.forwardProjection_plan_id(fixedIndex), fixedTransform, [], obj.Param.NumImages);
                    else
                        obj.Param.FixedVolumesDRR = obj.regTools.ForwardProject(obj.Param.forwardProjection_plan_id(fixedIndex), fixedTransform, [], obj.Param.NumImages, [], [], obj.regTools.MemoryStoreMode_Additive);
                    end
                end
                
                % copy the fixed DRR to all gpus
                if numFixedVolumes > 0
                    obj.regTools.SetInitialProjectionOnDevice(obj.Param.FixedVolumesDRR);
                end
                
                if(i==1)
                    % show initial guess
                    obj.Param.display_result = true;
                    obj.Param.current_iteration = -1;
                    obj.Param.CurrentStartPose = zeros(1,length(obj.InitialGuess));
                    feval(obj.ObjectiveFunctionName, zeros(1,length(obj.InitialGuess)), 1, obj.regTools, obj.Param);
                    obj.Param.display_result = false;
                end
                
                if(strcmp(obj.Optimizer, 'CMA-ES-PARALLEL') || strcmp(obj.Optimizer, 'CMA-ES'))
                    if(iscell(obj.UBounds))
                        % each cell element represents UBounds and LBounds
                        % of each start (dimension x num_multi_start)
                        cmaes_opt.UBounds = obj.UBounds{i};
                        cmaes_opt.LBounds = obj.LBounds{i};
                        in_sigmas = obj.InSigmas{i};
                    elseif(size(obj.UBounds,2)>=i)
                        cmaes_opt.UBounds = repmat(obj.UBounds(:,i),1,num_current_multi_start);
                        cmaes_opt.LBounds = repmat(obj.LBounds(:,i),1,num_current_multi_start);
                        in_sigmas = obj.InSigmas(:,i);
                    else
                        cmaes_opt.UBounds = Inf(size(obj.Param.CurrentStartPose));
                        cmaes_opt.LBounds = Inf(size(obj.Param.CurrentStartPose));
                        in_sigmas = obj.InSigmas(:,i);
                    end
                end
                
                if(strcmp(obj.Optimizer, 'CMA-ES-PARALLEL'))
                    % run parallel CMA-ES optimizer
                    cmaes_opt.MaxFunEvals = obj.MaxEval(i);
                    cmaes_opt.EvalParallel = true;
                    cmaes_opt.PopSize = popSize;
                    cmaes_opt.DispModulo = 0;
                    cmaes_opt.LogModulo = 0;
                    cmaes_opt.LogTime = 0;
                    cmaes_opt.SaveVariables = 'off';
                    cmaes_opt.DispFinal = 'off';
                    cmaes_opt.LogPlot = 'off';
                    cmaes_opt.ReadSignals = 0;
                    cmaes_opt.EvalInitialX = 0;
                    if(~isempty(obj.TolX)), cmaes_opt.TolX = obj.TolX(i); end
                    if(~isempty(obj.TolFun)), cmaes_opt.TolFun = obj.TolFun(i); end
                    cmaes_opt.Seed = obj.RandSeed;
                    obj.Param.CurrentStartPose = start_guesses;
                    [searched_multi_start, fmins, counteval, stopflag, out, bestever, cmaes_sigmas, cmaes_Bs, cmaes_diagDs, cmaes_C] = ...
                        cmaes_ex_par( obj.ObjectiveFunctionName, ...
                            zeros(size(obj.Param.CurrentStartPose')), ...  % objective variables initial point, determines N
                            in_sigmas, ... % initial coordinate wise standard deviation(s)
                            cmaes_opt, ...                                         % options struct, see defopts below
                            obj.regTools, ...
                            obj.Param ...
                        );
                    searched = searched_multi_start;
                    
                    if(obj.Debug)
                        for current = 1:size(searched_multi_start,2)
                            % show current progress
                            obj.Param.display_result = true;
                            feval(obj.ObjectiveFunctionName, searched_multi_start(:,current), current, obj.regTools, obj.Param);
                            obj.Param.display_result = false;
                        end
                    end
                else
                    for j=1:num_current_multi_start
                        if(i==1)
                            obj.Param.current_start = j;
                            obj.Param.CurrentStartPose = start_guesses(j,:);
                        end
                        currentGuess = zeros(1,size(start_guesses,2));
%                         if(i==1)
%                             currentGuess = start_guesses(j,:);
%                         end

%                         if(size(obj.UBounds,2)>=i)
%                             UB = obj.UBounds(:,i);
%                             LB = obj.LBounds(:,i);
%                         else
%                             UB = ones(length(currentGuess),1) * inf;
%                             LB = ones(length(currentGuess),1) *-inf;
%                         end
                        if(strcmp(obj.Optimizer, 'None')),
                            searched = currentGuess;
                            obj.fmin = 0;
                        elseif(strcmp(obj.Optimizer, 'CMA-ES'))
                            % run CMA-ES optimizer
                            cmaes_opt.MaxFunEvals = obj.MaxEval(i);
                            if(obj.SerialCMAES)
                                cmaes_opt.EvalParallel = false;
                                if(strcmp(obj.ObjectiveFunctionName,'Registration2D3D_IntensityBased.ComputeObjectiveFunction'))
                                    obj.ObjectiveFunctionName = 'Registration2D3D_IntensityBased.ComputeObjectiveFunction_Serial';
                                end
                            else
                                cmaes_opt.EvalParallel = true;
                            end
                            cmaes_opt.PopSize = popSize;
                            cmaes_opt.DispModulo = 0;
                            cmaes_opt.LogModulo = 0;
                            cmaes_opt.LogTime = 0;
                            cmaes_opt.SaveVariables = 'off';
                            cmaes_opt.DispFinal = 'off';
                            cmaes_opt.LogPlot = 'off';
                            cmaes_opt.ReadSignals = 0;
                            cmaes_opt.EvalInitialX = 0;
%                             if(size(obj.UBounds,2)>=i)
%                                 cmaes_opt.UBounds = UB;
%                                 cmaes_opt.LBounds = LB;
%                             end
                            if(~isempty(obj.TolX)), cmaes_opt.TolX = obj.TolX(i); end
                            if(~isempty(obj.TolFun)), cmaes_opt.TolFun = obj.TolFun(i); end
                            cmaes_opt.Seed = obj.RandSeed;
                            if(obj.SerialCMAES)
                                [searched, fmins(j), counteval, stopflag, out, bestever] = ...
                                    cmaes_ex( obj.ObjectiveFunctionName, ...
                                        currentGuess', ...  % objective variables initial point, determines N
                                        in_sigmas(:,j), ... % initial coordinate wise standard deviation(s)
                                        cmaes_opt, ...                                         % options struct, see defopts below
                                        obj.regTools, ...
                                        obj.Param ...
                                    );
                            else
                                [searched, fmins(j), counteval, stopflag, out, bestever, cmaes_sigmas(i), cmaes_Bs(:,:,i), cmaes_diagDs(:,i), cmaes_C(:,:,i)] = ...
                                    cmaes_ex_par( obj.ObjectiveFunctionName, ...
                                        currentGuess', ...  % objective variables initial point, determines N
                                        in_sigmas(:,j), ... % initial coordinate wise standard deviation(s)
                                        cmaes_opt, ...                                         % options struct, see defopts below
                                        obj.regTools, ...
                                        obj.Param ...
                                    );
                            end
%                             obj.DiagD = diagD_final;
%                             obj.Median_m_Best = median_m_best;
    %                         fprintf('final standard deviation: (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f), ratio: %f, median_m_best: %f\n', ...
    %                             std_dev_final, obj.DiagD, obj.Median_m_Best);

                        elseif(strcmp(obj.Optimizer, 'ga'))
                            options = gaoptimset('Vectorized','on','Display','diagnose','PopulationSize',popSize,'PopInitRange',[LB'; UB'] ...
                                ,'StallGenLimit',5);
                            f = str2func(['@(x, recon, param)' obj.ObjectiveFunctionName '(x, recon, param)']);
                            [searched, fmins(j)] = ga( {f, obj.regTools, obj.Param}, length(currentGuess), ...
                                [],[],[],[],LB,UB,[],options);

                        elseif(strcmp(obj.Optimizer, 'patternsearch'))
                            options = psoptimset('CompletePoll','on','Vectorized','on','Display','iter','InitialMeshSize',obj.Param.MaxParallel,'UseParallel','always');
                            f = str2func(['@(x, recon, param)' obj.ObjectiveFunctionName '(x, recon, param)']);
                            [searched, fmins(j)] = patternsearch( {f, obj.regTools, obj.Param}, currentGuess, ...
                                [],[],[],[],LB,UB,[],options);

                        elseif(strcmp(obj.Optimizer, 'fminsearch'))
                            if(~isempty(obj.TolX)), tolX = obj.TolX(i); else tolX = 1e-2; end
                            if(~isempty(obj.TolFun)), tolFun = obj.TolFun(i); else tolFun = 1e+2; end
                            % Simplex method
                            [searched, fmins(j)] = fminsearch_ex( str2func(['@(x, restart_indices, reg, param) ' obj.ObjectiveFunctionName '(x, restart_indices, reg, param)']), currentGuess, ...
                                optimset( 'TolX', tolX, 'TolFun', tolFun, 'MaxFunEvals', obj.MaxEval(i) ), ...
                                obj.fminsearch_initial_size(i,:), obj.fminsearch_initial_size(i,:), 1, obj.regTools, obj.Param );
                        end

                        if(obj.Param.RecordLog), obj.Param.RegistrationLogs_LevelEnd(i,1) = size( obj.Param.RegistrationLogs, 1 ); end
%                         if(i==1)
                            searched_multi_start(:,j) = searched(:);
%                         end
                                
                        % show current progress
                        obj.Param.display_result = true;
                        if(obj.SerialCMAES)
                            feval(obj.ObjectiveFunctionName, searched(:), obj.regTools, obj.Param);
                        else
                            feval(obj.ObjectiveFunctionName, searched(:), 1, obj.regTools, obj.Param);
                        end
                        obj.Param.display_result = false;
                    end
                end
                
                obj.AllMultistartSolutions = zeros(num_current_multi_start,size(start_guesses,2));
                
                %{
                if(~all(obj.Param.PerturbationCenter==0))
                    % add perturbation center offset if needed
                    searched_multi_start(1:6,:) = RegTools.PrePostMultiplyTranslation(searched_multi_start(1:6,:), obj.Param.PerturbationCenter); 
                end
                %}
                
                for j=1:num_current_multi_start
                    % compute start pose of next level
%                     if (strcmp(obj.ObjectiveFunctionName, 'Registration2D3D_IntensityBased.ObjectiveFunction_Multiple'))
                        numMovingVolumes = length(obj.Param.MovingVolumesIndices);
                        for k = 1:numMovingVolumes
                            resultOffset = (k-1)*6;
                            obj.AllMultistartSolutions(j,resultOffset+(1:6)) = RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( ...
                                                        searched_multi_start(resultOffset+(1:6),j), RegTools.convertTransRotTo4x4(start_guesses(j,resultOffset+(1:6))), ...
                                                        obj.Param.OffsetMultiplicationOrder ) );
                        end
%                     else
%                         obj.AllMultistartSolutions(j,:) = [ RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( ...
%                                                     searched_multi_start(1:6,j), RegTools.convertTransRotTo4x4(start_guesses(j,1:6)), ...
%                                                     obj.Param.OffsetMultiplicationOrder ) ), searched_multi_start(7:end,j)'+start_guesses(j,7:end) ];
%                     end
                end
                
                
                obj.fmin = min(fmins);
                obj.SuccessfulStart = find(fmins == obj.fmin, 1, 'first');
                if(isempty(obj.SuccessfulStart)), obj.SuccessfulStart = 1; end
%                 if(obj.MultistartCMAES_myu<0)
                    obj.CMAES_sigma = cmaes_sigmas(obj.SuccessfulStart);
                    obj.CMAES_B = cmaes_Bs(:,:,obj.SuccessfulStart);
                    obj.CMAES_diagD = cmaes_diagDs(:,obj.SuccessfulStart);
                    obj.CMAES_C = cmaes_C(:,:,obj.SuccessfulStart);
                
%                     if(i==1)
                        % compute start pose of next level
                        obj.Param.CurrentStartPose = obj.AllMultistartSolutions(obj.SuccessfulStart,:);
                if(~all(obj.Param.PerturbationCenter==0))
                    % add perturbation center offset if needed
%                     if (strcmp(obj.ObjectiveFunctionName, 'Registration2D3D_IntensityBased.ObjectiveFunction_Multiple'))
                        numMovingVolumes = length(obj.Param.MovingVolumesIndices);
                        for k = 1:numMovingVolumes
                            offset = (k-1)*6;
                            obj.AllMultistartSolutions(:,offset+(1:6)) = RegTools.ChangeRotationCenter(obj.AllMultistartSolutions(:,offset+(1:6))', obj.Param.PerturbationCenter(:,k)' )' ; 
                        end
%                     else
%                         obj.AllMultistartSolutions(:,1:6) = RegTools.ChangeRotationCenter(obj.AllMultistartSolutions(:,1:6)', obj.Param.PerturbationCenter)'; 
%                     end
                end
%                     end
%                 else
%                     [sorted, indx] = sort(fmins);
%                     numSelected = floor(length(fmins)*obj.MultistartCMAES_myu);
%                     obj.CMAES_C = mean(cmaes_C(:,:,indx(1:numSelected)),3);
%                     [obj.CMAES_B,tmp] = eig(obj.CMAES_C);     % eigen decomposition, B==normalized eigenvectors
%                     obj.CMAES_diagD = sqrt(diag(tmp)); 
%                     obj.CMAES_sigma = mean(cmaes_sigmas(indx(1:numSelected)));
%                     obj.Param.CurrentStartPose = mean(obj.AllMultistartSolutions(indx(1:numSelected),1));
%                 end
                if(obj.Debug)
                    disp(['successful: ' num2str(obj.SuccessfulStart) ', fmins: ' num2str(fmins')]);
                    debug_Nx6 = zeros(num_current_multi_start,6);
                    for temp=1:num_current_multi_start
                        debug_Nx6(temp,:) = RegTools.convert4x4ToTransRot( ...
                            RegTools.convertTransRotTo4x4(searched_multi_start(:,temp)) * ...
                            RegTools.convertTransRotTo4x4(start_guesses(temp,:)) );
                    end
                    disp(num2str(debug_Nx6));
                end
                
                start_guesses = obj.Param.CurrentStartPose;
                
                % show current progress
                obj.Param.display_result = true;
                if(obj.SerialCMAES)
                    feval(obj.ObjectiveFunctionName, zeros(1,size(start_guesses,2)), obj.regTools, obj.Param);
                else
                    feval(obj.ObjectiveFunctionName, zeros(1,size(start_guesses,2)), 1, obj.regTools, obj.Param);
                end
                obj.Param.display_result = false;
                obj.Param.CurrentStartPose = obj.AllMultistartSolutions(obj.SuccessfulStart,:);

                % clean up (for each level)
                obj.regTools.DeleteProjectionParametersArray( geomID );
                obj.regTools.DeleteSimilarityMeasureComputationPlan( obj.Param.similarity_measure_plan_id );
                for j=1:size(obj.all_volumes,2)
                    obj.regTools.DeleteForwardProjectionPlan( obj.Param.forwardProjection_plan_id(j) );
                end
                obj.RegistrationTime(i) = toc(obj.Param.TimerID) + obj.Param.initialSearchTime;
                obj.NumIterations(i) = obj.Param.current_iteration;
                if(i>1), 
                    obj.RegistrationTime(i) = obj.RegistrationTime(i)-sum(obj.RegistrationTime(1:(i-1))); 
                    obj.NumIterations(i) = obj.NumIterations(i)-sum(obj.NumIterations(1:(i-1))); 
                end
                
                % add initial guess offset and store as an intermediate result
%                 if (strcmp(obj.ObjectiveFunctionName, 'Registration2D3D_IntensityBased.ObjectiveFunction_Multiple'))
                        numMovingVolumes = length(obj.Param.MovingVolumesIndices);
                        for j = 1:numMovingVolumes
                            resultOffset = (j-1)*6;
                            if(strcmp(obj.Param.OffsetMultiplicationOrder,'Post'))
                                volume_offset_4x4 = RegTools.matTranslation(obj.Param.VolumeOffsets(:,k));
                                initial_guess_global = RegTools.convertTransRotTo4x4(obj.Param.InitialGuess(resultOffset+(1:6))) * volume_offset_4x4;
                                obj.IntermediateTransform(i,resultOffset+(1:6)) = RegTools.convert4x4ToTransRot( initial_guess_global * ...
                                    RegTools.convertTransRotTo4x4_multi(obj.Param.CurrentStartPose(resultOffset+(1:6))) * inv(volume_offset_4x4) );
                            else
                                obj.IntermediateTransform(i, resultOffset+(1:6)) = RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( ...
                                                                obj.Param.CurrentStartPose(resultOffset+(1:6)), RegTools.convertTransRotTo4x4(obj.Param.InitialGuess(resultOffset+(1:6))), ...
                                                                obj.Param.OffsetMultiplicationOrder ) );
                            end
                        end
%                 else
%                     obj.IntermediateTransform(i,:) = [RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( ...
%                                                             obj.Param.CurrentStartPose(1:6), RegTools.convertTransRotTo4x4(obj.Param.InitialGuess(1:6)), ...
%                                                             obj.Param.OffsetMultiplicationOrder ) ), obj.Param.CurrentStartPose(7:end)+obj.Param.InitialGuess(7:end)'];
%                 end
                if(i==1)
                    obj.AllMultistartFmins = fmins;
                    if(~isempty(obj.SearchRange))
                        ranges = ones(num_current_multi_start,1) * obj.SearchRange;
                        out_of_bound = any(obj.AllMultistartSolutions < -ranges | obj.AllMultistartSolutions > ranges, 2);
                        obj.AllMultistartFmins(out_of_bound) = 0.0;
                    end
                    
                    for j=1:num_current_multi_start
%                         if (strcmp(obj.ObjectiveFunctionName, 'Registration2D3D_IntensityBased.ObjectiveFunction_Multiple'))
                            numMovingVolumes = length(obj.Param.MovingVolumesIndices);
                            for k = 1:numMovingVolumes
                                resultOffset = (k-1)*6;
                                if(strcmp(obj.Param.OffsetMultiplicationOrder,'Post'))
                                    volume_offset_4x4 = RegTools.matTranslation(obj.Param.VolumeOffsets(:,k));
                                    initial_guess_global = RegTools.convertTransRotTo4x4(obj.Param.InitialGuess(resultOffset+(1:6))) * volume_offset_4x4;
                                    obj.AllMultistartSolutions(j,resultOffset+(1:6)) = RegTools.convert4x4ToTransRot( initial_guess_global * ...
                                        RegTools.convertTransRotTo4x4_multi(obj.AllMultistartSolutions(j,resultOffset+(1:6))) * inv(volume_offset_4x4) );
                                else
                                    obj.AllMultistartSolutions(j,resultOffset+(1:6)) = RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( ...
                                        obj.AllMultistartSolutions(j,resultOffset+(1:6)), ...
                                        RegTools.convertTransRotTo4x4(obj.Param.InitialGuess(resultOffset+(1:6))), ...
                                        obj.Param.OffsetMultiplicationOrder) );
                                end
                            end
%                         else
%                             obj.AllMultistartSolutions(j,:) = [RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( ...
%                                                              obj.AllMultistartSolutions(j,1:6), RegTools.convertTransRotTo4x4(obj.Param.InitialGuess(1:6)), ...
%                                                              obj.Param.OffsetMultiplicationOrder) ), obj.AllMultistartSolutions(j,7:end) + obj.Param.InitialGuess(7:end)'];
%                         end
                    end
                end
            end
            obj.RegistrationTransform = obj.IntermediateTransform(end,:);
            obj.RegistrationTransform_4x4 = obj.regTools.convertTransRotTo4x4( obj.RegistrationTransform(1:6) );
        end % Run
        
        function freeMem_byte = GPUmemCheck( obj, varargin )
            freeMem_byte = obj.regTools.GPUmemCheck( varargin{:} );
        end
                        
        function [ObjectiveValues, AllImages, SMImages, GradImages, DiffImages] = FailureAnalysis(obj, true_vec, failed_vec, numSamplePoints, start_stop, similarity_measure)
            start_pos = failed_vec + (true_vec-failed_vec) * start_stop(1);
            stop_pos = failed_vec + (true_vec-failed_vec) * start_stop(2);
            step = (stop_pos-start_pos) / numSamplePoints;
            
            DS_l = obj.InitialSearch_DownsampleRatio(1); DS_u = obj.InitialSearch_DownsampleRatio(2);
            img_dim = size(obj.initial_search_image);
            obj.regTools.SetTransferBlockSize(numSamplePoints);
            obj.regTools.SetStepSize(mean(obj.VoxelSize*DS_u)*2);
            forwardProjection_plan_id = obj.regTools.CreateForwardProjectionPlan( obj.initial_search_volume,  obj.VoxelSize*DS_u );
            if(length(true_vec)==9)
                ProjectionMatrices = zeros(3, 4, numSamplePoints);
                for i=1:numSamplePoints
                    ProjectionMatrices(:,:,i) = RegTools.generatePM_FromSourcePositions(start_pos(7:9) + (i-1)*step(7:9), img_dim, obj.PixelSize*DS_l);
                end
                geomID = obj.regTools.GenerateGeometry_3x4ProjectionMatrix( ProjectionMatrices, obj.PixelSize*DS_l, img_dim );
            else
                ProjectionMatrices = repmat(obj.ProjectionMatrices(:,:,1), [1 1 numSamplePoints]);
                geomID = obj.regTools.GenerateGeometry_3x4ProjectionMatrix( ProjectionMatrices, obj.PixelSize*DS_l, img_dim, DS_l );
            end

            similarity_measure_plan_id = obj.CreateSimilarityMeasureComputationPlan( similarity_measure, obj.initial_search_image, obj.initial_search_mask, numSamplePoints );
            if(nargout>=4)
                GradImages = zeros([img_dim, 3]);
                GradImages(:,:,1) = obj.regTools.GetSimilarityMeasureComputationPlan(similarity_measure_plan_id, 2);
                GradImages(:,:,2) = obj.regTools.GetSimilarityMeasureComputationPlan(similarity_measure_plan_id, 3);
                GradImages(:,:,3) = sqrt(GradImages(:,:,1).^2+GradImages(:,:,2).^2);
                if(~isempty(obj.SimilarityMeasureMask))
                    for i=1:3, GradImages(:,:,i) = GradImages(:,:,i) .* obj.initial_search_mask; end
                end
            end
            F_fixed_floating4x4 = RegTools.convertTransRotTo4x4_multi( start_pos(1:6)'*ones(1,numSamplePoints)+step(1:6)'*([1:numSamplePoints]-1) );
            if(nargout>=3)
                % render each DRR separately to get SMImages for each DRR
                AllImages = zeros([img_dim, numSamplePoints]);
                ObjectiveValues = zeros([numSamplePoints 1]);
                if(similarity_measure == RegTools.SimilarityMeasureType_MI || similarity_measure == RegTools.SimilarityMeasureType_NMI)
                    SMImages = zeros([obj.MI_NumberOfBins obj.MI_NumberOfBins numSamplePoints]);
                else
                    SMImages = zeros([img_dim, numSamplePoints]);
                end
                for i=1:numSamplePoints
                    obj.regTools.SetProjectionParameter_3x4ProjectionMatrix( 0, ProjectionMatrices(:,:,i)', obj.PixelSize*DS_l, img_dim );
                    AllImages(:,:,i) = obj.regTools.ForwardProject(forwardProjection_plan_id, F_fixed_floating4x4(:,:,i), [], 1);
                    ObjectiveValues(i) = obj.regTools.ComputeSimilarityMeasure( similarity_measure_plan_id, similarity_measure, 1 )';
                    if(similarity_measure == RegTools.SimilarityMeasureType_MI || similarity_measure == RegTools.SimilarityMeasureType_NMI)
                        % get joint histogram
                        SMImages(:,:,i) = obj.regTools.GetSimilarityMeasureComputationPlan(similarity_measure_plan_id, 9, -1, obj.MI_NumberOfBins);
                    else
                        SMImages(:,:,i) = obj.regTools.GetSimilarityMeasureComputationPlan(similarity_measure_plan_id, 1);
                    end
                end
            else
                if(nargout==2),
                    AllImages = obj.regTools.ForwardProject(forwardProjection_plan_id, F_fixed_floating4x4, [], 1);
                else
                    obj.regTools.ForwardProject(forwardProjection_plan_id, F_fixed_floating4x4, [], 1);
                end
                ObjectiveValues = obj.regTools.ComputeSimilarityMeasure( similarity_measure_plan_id, similarity_measure, numSamplePoints ); % cost: column vector
            end
            if(nargout>=5)
                DiffImages = zeros([img_dim, numSamplePoints]);
                % return difference images
                for i=1:numSamplePoints
                    % intensity (window/level) adjustment
%                     l = obj.least_squares_line([obj.initial_search_image(:) reshape(AllImages(:,:,i),[],1)]);
%                     a = -l(1)/l(2); b = -l(3)/l(2); % y = ax + b
%                     DiffImages(:,:,i) = (AllImages(:,:,i)/a - b/a) - obj.initial_search_image;
                    DiffImages(:,:,i) = PointwiseMutualInformation(AllImages(:,:,i), obj.initial_search_image, 512, 1);
                end
            end
            obj.regTools.DeleteSimilarityMeasureComputationPlan( similarity_measure_plan_id );
            obj.regTools.DeleteProjectionParametersArray( geomID );
            obj.regTools.DeleteForwardProjectionPlan( forwardProjection_plan_id );
        end
        
        function [estimated_6vec, time, similarity_vec, DRRs] = InitialSearch(obj, pm_base, extrinsic, search_trans_6vec, pre_or_post, max_parallel_eval)
            tic;
            numEval = size(search_trans_6vec,1);
            if(~exist('max_parallel_eval','var')), max_parallel_eval = 1000; end
            if(~exist('pre_or_post','var')), pre_or_post = 'Pre'; end
            pEval = min(max_parallel_eval, numEval);     % number of parallel evaluations
            
            % forward projection of all views (number of row of
            % 'search_trans_6vec' should not exceed maximum transfer block size (default: 800)
            DS_l = obj.InitialSearch_DownsampleRatio(1); DS_u = obj.InitialSearch_DownsampleRatio(2);
            obj.regTools.SetVolumeInfo( struct('VolumeDim', size(obj.initial_search_volume), 'VoxelSize', obj.VoxelSize*DS_u) );
            forwardProjection_plan_id = obj.regTools.CreateForwardProjectionPlan( obj.initial_search_volume,  obj.VoxelSize*DS_u );
            similarity_measure_plan_id = obj.regTools.CreateSimilarityMeasureComputationPlan( obj.initial_search_image, obj.GI_Sigma, [], pEval ); % don't use LogLikelihood
            img_dim = size(obj.initial_search_image);
            geomID = obj.regTools.GenerateGeometry_3x4ProjectionMatrix( repmat(pm_base,[1 1 pEval]), obj.PixelSize*DS_l, img_dim(1:2), DS_l );
            obj.regTools.SetTransferBlockSize(pEval);
            obj.regTools.SetStepSize(mean(obj.VoxelSize*DS_u)*2);
            obj.regTools.SetProjectorMode(obj.ProjectorMode);
            obj.regTools.SetRayCastingThreshold(obj.RayCastingThreshold);
            
            if(nargout>=4), DRRs = zeros([obj.regTools.GetProjectionDim() numEval]); end
            numLoop = ceil(numEval/pEval);
            similarity_vec = zeros(1,numEval);
            for loop = 1:numLoop
                if(loop<numLoop), n_eval = pEval; else n_eval = numEval-(numLoop-1)*pEval; end
                freeMem_byte = obj.GPUmemCheck('Optimization start');
                fprintf('computing sample %d to %d (%f MB available on the GPU)\n', (loop-1)*pEval, (loop-1)*pEval+n_eval, freeMem_byte/1024/1024);
                search_trans_4x4 = RegTools.convertTransRotTo4x4_multi( search_trans_6vec((1:n_eval)+(loop-1)*pEval,:)', extrinsic, pre_or_post );
                if(nargout>=4)
                    DRRs(:,:,(1:n_eval)+(loop-1)*pEval) = obj.regTools.ForwardProject(forwardProjection_plan_id, search_trans_4x4, [], 1);
                else
                    obj.regTools.ForwardProject(forwardProjection_plan_id, search_trans_4x4, [], 1);
                end
                similarity_vec(1,(1:n_eval)+(loop-1)*pEval) = obj.regTools.ComputeSimilarityMeasure( similarity_measure_plan_id, RegTools.SimilarityMeasureType_GI_SINGLE, n_eval )'; % cost: row vector
            end
            
            % clean up
            obj.regTools.DeleteSimilarityMeasureComputationPlan( similarity_measure_plan_id );
            obj.regTools.DeleteForwardProjectionPlan( forwardProjection_plan_id );
            obj.regTools.DeleteProjectionParametersArray( geomID );

            % return transformation that achieved maximum similarity
            estimated_6vec = search_trans_6vec(find(similarity_vec == max(similarity_vec(:)), 1, 'first'),:);
            time = toc;
        end
        
        function SetDownsampleRatioByPyramidResolutions( obj, pyramid_resolutions, magnification )
            % determine downsampling ratio based on size of the images
            obj.DownsampleRatio_Projection = sqrt(numel(obj.GetImages())./(pyramid_resolutions.^2));  % downsample so that the area becomes (res.^2)
            obj.DownsampleRatio_Volume = cell(length(obj.DownsampleRatio_Projection),1);
            for level=1:length(obj.DownsampleRatio_Projection)
                obj.DownsampleRatio_Volume{level} = max(obj.DownsampleRatio_Projection(level)*max(obj.PixelSize(:))/magnification./obj.VoxelSize, 1);         % voxel_size ~= pixel_size/magnification
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%
    % Static methods %
    %%%%%%%%%%%%%%%%%%
    methods (Static)
        function cost = ComputeObjectiveFunction_Row(p_cur, regTools, param)
            cost = Registration2D3D_IntensityBased.ComputeObjectiveFunction(p_cur', regTools, param);
        end
        
        function cost = ComputeObjectiveFunction_Serial(p_cur, regTools, param)
            cost = Registration2D3D_IntensityBased.ComputeObjectiveFunction(p_cur, 1, regTools, param);
        end
        
        function cost = ComputeObjectiveFunction(p_cur, restart_indices, regTools, param)
            % rows 1-6 of p_cur are for rigid transformation, 7-9 for source position if isEstimateSource is true.
            if(isvector(p_cur)), p_cur = p_cur(:); end
            [N, Q, M] = size(p_cur);
        %             disp(['restart_indices: ' num2str(restart_indices')]);
            pEval = Q*M;
            p_cur = reshape(p_cur, N, Q*M);
            if(~all(param.PerturbationCenter==0))
                p_cur(1:6,:) = reshape( RegTools.PrePostMultiplyTranslation(p_cur(1:6,:), param.PerturbationCenter), [6 Q*M] );
%                 p_cur(1:6,:) = reshape( RegTools.ChangeRotationCenter(p_cur(1:6,:), param.PerturbationCenter), [6 Q*M] ); 
            end
            current_start_poses4x4 = RegTools.convertTransRotTo4x4_multi(param.CurrentStartPose(restart_indices,1:6)', RegTools.convertTransRotTo4x4(param.InitialGuess(1:6)'), param.OffsetMultiplicationOrder);
            current_start_poses4x4_all = reshape( permute( repmat(current_start_poses4x4, [1 1 1 Q]), [1 2 4 3] ), 4, 4, []);
            if(strcmp(param.OffsetMultiplicationOrder, 'Pre'))
                F_fixed_floating4x4 = RegTools.MultiProd(current_start_poses4x4_all, RegTools.convertTransRotTo4x4_multi(p_cur(1:6,:)));
            else
                F_fixed_floating4x4 = RegTools.MultiProd(RegTools.convertTransRotTo4x4_multi(p_cur(1:6,:)), current_start_poses4x4_all);
            end
            p_cur(7:end,:) = p_cur(7:end,:) + reshape( repmat( param.CurrentStartPose(restart_indices,7:end)', [Q 1]), [], Q*M);
            if(length(param.InitialGuess)>6)
                p_cur(7:end,:) = p_cur(7:end,:) + param.InitialGuess(7:end)*ones(1,Q*M);
            end

            % divide the renderings into subset in case population size excess the maximum available parallel rendering
            numLoop = ceil(pEval/param.MaxParallel);
            step = min(pEval, param.MaxParallel);

            cost = zeros(1,pEval);
            for loop = 1:numLoop
                if(loop<numLoop), numEval = step; else numEval = pEval-(numLoop-1)*step; end

                if(param.isEstimateSourcePosition)
                    % if isEstimateSource option is on, set projection matrices using the estimated source positions
                    uv_dim = regTools.GetProjectionDim();
                    PixelSize = regTools.GetPixelSize();
                    ProjectionMatrices = RegTools.generatePM_FromSourcePositions(p_cur(7:9,:)', uv_dim, PixelSize);
                    regTools.SetProjectionParameter_3x4ProjectionMatrix_Multi( pEval, libpointer( 'doublePtr', permute(ProjectionMatrices,[2 1 3]) ), PixelSize, uv_dim);
                end
                % forward project and compute similarity measure
                regTools.ForwardProject(param.forwardProjection_plan_id, F_fixed_floating4x4(:,:,(1:numEval)+(loop-1)*step), [], param.NumImages);
                cost(1,(1:numEval)+(loop-1)*step) = -regTools.ComputeSimilarityMeasure( param.similarity_measure_plan_id, param.SimilarityMeasureType, numEval )'; % cost: row vector
                [current_min_cost, similarity_computation_frame] = min(cost(1,(1:numEval)+(loop-1)*step)); % the frame number corresponds to the image that GetSimilarityMeasureComputationPlan() will return (for progress monitoring)
                similarity_computation_frame_4x4 = F_fixed_floating4x4(:,:,similarity_computation_frame+(loop-1)*step);
                current_p = p_cur(:,similarity_computation_frame+(loop-1)*step);
                current = current_p;    % this is just for backward compatibility (will be removed in the future)
%                 similarity_computation_frame = 1+(loop-1)*step; % the frame number corresponds to the image that GetSimilarityMeasureComputationPlan() will return (for progress monitoring)
            end
            cost(cost==0) = Inf;
            cost(isnan(cost)) = 0;

            if(param.RecordLog)
                % add current results to RegistrationLog
                p_log = p_cur';
                param.RegistrationLogs = cat(1, param.RegistrationLogs, [p_log cost' repmat(toc(param.TimerID), [pEval 1])]);
            end

            % show progress if needed
            if(param.current_iteration >= param.NextProgressShow || toc(param.TimerID) >= param.NextProgressShowTime || param.display_result)
%                 current = [RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( p_cur(1:6,end), current_start_poses4x4(:,:,end) ) ), p_cur(7:end,end)'];
%                 if(param.RecordLog), 
%                     current = p_log(similarity_computation_frame,:)';
%                 else
%                     current = p_cur(:,similarity_computation_frame);
%                 end
                if(~isempty(param.ShowProgressFunc)), eval([param.ShowProgressFunc ';']); end
                while(param.current_iteration >= param.NextProgressShow || toc(param.TimerID) >= param.NextProgressShowTime)
                    param.NextProgressShow = param.NextProgressShow + param.ProgressShowStep;
                    param.NextProgressShowTime = param.NextProgressShowTime + param.ProgressShowInterval_sec;
                end
            end

            param.current_iteration = param.current_iteration + pEval;
            cost = reshape(cost, Q, [])';
        end
        
        
        function cost = ObjectiveFunction_Multiple(p_cur, restart_indices, regTools, param)
            % rows 1-6xN of p_cur are for rigid transformation
            if(isvector(p_cur)), p_cur = p_cur(:); end
            [N, Q, M] = size(p_cur);
            % N: Total number of rigid transform parameters for all objects
            % Q: PopSize
            % M: Number of multi starts

            fixedVolumesIndices = param.FixedVolumesIndices;
            movingVolumesIndices = param.MovingVolumesIndices;
            numFixedVolumes = length(fixedVolumesIndices);
            numMovingVolumes = length(movingVolumesIndices);

            pEval = Q*M;
            p_cur = reshape(p_cur, N, Q*M);
            
            F_fixed_floating4x4 = zeros(4, 4, pEval, numMovingVolumes);
            for i = 1:numMovingVolumes
                offset = (i-1)*6;
%                 current_start_poses4x4 = RegTools.convertTransRotTo4x4_multi(param.CurrentStartPose(restart_indices, offset+(1:6))', ...
%                     RegTools.convertTransRotTo4x4(param.InitialGuess(offset+(1:6))'), param.OffsetMultiplicationOrder);
                current_start_poses4x4 = RegTools.convertTransRotTo4x4_multi(param.CurrentStartPose(restart_indices, offset+(1:6))');
                current_start_poses4x4_all = reshape( permute( repmat(current_start_poses4x4, [1 1 1 Q]), [1 2 4 3] ), 4, 4, []);
                initial_guess4x4 = RegTools.convertTransRotTo4x4(param.InitialGuess(offset+(1:6)));
                initial_guess4x4_all = repmat( initial_guess4x4, [1 1 length(restart_indices)*Q] );
                p_cur_all = RegTools.convertTransRotTo4x4_multi(p_cur(offset+(1:6),:));
                
                if(strcmp(param.OffsetMultiplicationOrder, 'Pre'))
                    % not tested
                    F_fixed_floating4x4(:,:,:,i) = RegTools.MultiProd(initial_guess4x4_all, current_start_poses4x4_all);
                    F_fixed_floating4x4(:,:,:,i) = RegTools.MultiProd(F_fixed_floating4x4(:,:,:,i), p_cur_all);
                else
                    % tested with single object
                    volume_offset_4x4 = RegTools.matTranslation(param.VolumeOffsets(:,i));
                    initial_guess_global = RegTools.convertTransRotTo4x4(param.InitialGuess(offset+(1:6))) * volume_offset_4x4;
                    initial_guess_global_all = repmat( initial_guess_global, [1 1 length(restart_indices)*Q] );
                    volume_offsets_all = repmat( inv(volume_offset_4x4), [1 1 length(restart_indices)*Q] );
                    F_fixed_floating4x4(:,:,:,i) = RegTools.MultiProd(p_cur_all, current_start_poses4x4_all);
                    F_fixed_floating4x4(:,:,:,i) = RegTools.PrePostMultiplyTranslation_4x4( F_fixed_floating4x4(:,:,:,i), param.PerturbationCenter );
                    F_fixed_floating4x4(:,:,:,i) = RegTools.MultiProd(F_fixed_floating4x4(:,:,:,i), volume_offsets_all); % multiply from the right
                    F_fixed_floating4x4(:,:,:,i) = RegTools.MultiProd(initial_guess_global_all, F_fixed_floating4x4(:,:,:,i)); % multiply from the left
                end
            end

            cost = zeros(1, pEval);
            % divide the renderings into subset in case population size excess the maximum available parallel rendering
            numLoop = ceil(pEval/param.MaxParallel);
            step = min(pEval, param.MaxParallel);

            for loop = 1:numLoop
                if(loop<numLoop), numEval = step; else numEval = pEval-(numLoop-1)*step; end
                
                for i = 1:numMovingVolumes
                    % forward project and compute similarity measure
                    movingIndex = movingVolumesIndices(i);
                    if i == 1
                        regTools.ForwardProject(param.forwardProjection_plan_id(movingIndex), F_fixed_floating4x4(:, :, (1:numEval)+(loop-1)*step, i), [], param.NumImages, [], [], regTools.MemoryStoreMode_Replace, param.FixedVolumesDRR);
                    else
                        regTools.ForwardProject(param.forwardProjection_plan_id(movingIndex), F_fixed_floating4x4(:, :, (1:numEval)+(loop-1)*step, i), [], param.NumImages, [], [], regTools.MemoryStoreMode_Additive);
                    end
                end
                cost(1, (1:numEval)+(loop-1)*step) = -regTools.ComputeSimilarityMeasure( param.similarity_measure_plan_id, param.SimilarityMeasureType, numEval )'; % cost: row vector
        %       similarity_computation_frame = 1+(loop-1)*step; % the frame number corresponds to the image that GetSimilarityMeasureComputationPlan() will return (for progress monitoring)
            end

            % these variables are used in the dipslay callback script!
            [current_min_cost, similarity_computation_frame] = min(cost); % the frame number corresponds to the image that GetSimilarityMeasureComputationPlan() will return (for progress monitoring)
            cost(cost==0) = Inf;
            cost(isnan(cost)) = 0;

            if(param.RecordLog)
                % add current results to RegistrationLog
                p_log = p_cur';
                param.RegistrationLogs = cat(1, param.RegistrationLogs, [p_log cost' repmat(toc(param.TimerID), [pEval 1])]);
            end

            % show progress if needed
            if(param.current_iteration >= param.NextProgressShow || toc(param.TimerID) >= param.NextProgressShowTime || param.display_result)
        %                 current = [RegTools.convert4x4ToTransRot( RegTools.convertTransRotTo4x4_multi( p_cur(1:6,end), current_start_poses4x4(:,:,end) ) ), p_cur(7:end,end)'];
        %                 if(param.RecordLog), 
        %                     current = p_log(similarity_computation_frame,:)';
        %                 else
        %                     current = p_cur(:,similarity_computation_frame);
        %                 end
                if(~isempty(param.ShowProgressFunc)), eval([param.ShowProgressFunc ';']); end
                while(param.current_iteration >= param.NextProgressShow || toc(param.TimerID) >= param.NextProgressShowTime)
                    param.NextProgressShow = param.NextProgressShow + param.ProgressShowStep;
                    param.NextProgressShowTime = param.NextProgressShowTime + param.ProgressShowInterval_sec;
                end
            end

            param.current_iteration = param.current_iteration + pEval;
            cost = reshape(cost, Q, [])';
        end
        
        function SetOptimizationSettingFunction( reg2D3D_obj, setting_id )
            % an implementation example of this function
            % (user can create a separate .m file and specify the function
            % name as OptimizationSettingFunctionName)
            switch setting_id
                case 1
                    reg2D3D_obj.NumMultiLevel = 2;
                    reg2D3D_obj.MaxEval = [1e6, 1e6];
                    reg2D3D_obj.TolX = [3.0, 0.1];
                    reg2D3D_obj.InSigmas = [[120, 120, 120, 12, 12, 12]', [3, 3, 3, 3, 3, 3]'];
                    reg2D3D_obj.UBounds = [[inf, 300, inf, 30, 30, 30]', [5, 5, 5, 5, 5, 5]'];
                    reg2D3D_obj.LBounds = -reg2D3D_obj.UBounds;
                    reg2D3D_obj.PopSize = [400.0, 100.0];
                    reg2D3D_obj.MultiStart = [[0, 0, 0, 0, 0, 0]; [0, 150, 0, 0, 0, 0]];
                    reg2D3D_obj.DownsampleRatio_Projection = [8, 4];
                    reg2D3D_obj.DownsampleRatio_Volume = [2, 1];
            end
        end % SetOptimizationSetting
        
        function ProjectionMatrices = GetProjectionMatrices_SingleImage2D3D( uv_dim, pixel_size, param )
            ProjectionMatrices = RegTools.generatePM_FromSourcePositions(param.SourcePosition_mm, uv_dim, pixel_size);
            if(isfield(param, 'ExtParameterOffset')), ProjectionMatrices = ProjectionMatrices * param.ExtParameterOffset; end
        end
        
        function out_images = LogCorrection(in_images, I0)
            % execute 'Log correction'
            % (divide input images by I0 and return negative log)
            % inputs:
            %   in_images: W x H x N (3D array), N images
            %   I0: N element of 1D vector, I0 values for each image
            % output:
            %   out_images: log corrected images

            if(~isa(in_images,'single')), in_images = single(in_images); end
            numImages = size(in_images, 3);
            if(~exist('I0','var')), I0 = ones(size(in_images,3),1); end
            for i=1:numImages
                in_images(:,:,i) = in_images(:,:,i) ./ I0(i);
            end
            in_images(in_images <= 0) = 1;
            out_images = -log( in_images );
        end
        
        function [line] = least_squares_line(points)
            %Function to fit a least squares line to a set of input points in a 2D. A
            %direction of the line is first sought using PCA analysis. Depending on the
            %orientation of the line, either y = mx + c is used...or x=my+c is used.
            %The pi/4 mark is used as a decider. for each of these cases, the
            %parameters are computed using the typical least square method.
            
            %PCA analysis to get approximate direction.
            pts(:,1) = points(:,1) - mean(points(:,1));
            pts(:,2) = points(:,2) - mean(points(:,2));
            PCA = [ sum(pts(:,1).*pts(:,1)) sum(pts(:,1).*pts(:,2)) ; sum(pts(:,1).*pts(:,2)) sum(pts(:,2).*pts(:,2))];
            [V D] = eig(PCA);
            [tmp, ind] = max(abs(diag(D)));
            V = V(:,ind);

            %Depending on the case, compute the line parameters.
            if(abs(V(1)) >= abs(V(2)) )
                A = [points(:,1) ones(size(points,1),1)];
                x = inv(A'*A)*(A'*points(:,2));
                line(1) = x(1);
                line(2) = -1;
                line(3) = x(2);
                if( isnan(line(1)) )
                    display 'What the?'
                end
                if(line(3)>0)
                    line = -line;
                end
                line = line/norm(line(1:2));
            else
                A = [points(:,2) ones(size(points,1),1)];
                x = inv(A'*A)*(A'*points(:,1));
                line(1) = -1;
                line(2) = x(1);
                line(3) = x(2);
                if( isnan(line(1)) )
                    display 'What the?'
                end
                if(line(3)>0)
                    line = -line;
                end
                line = line/norm(line(1:2));
            end
        end
        
        function pMI = PointwiseMutualInformation(data1, data2, numBins, isNormalized)
            mpd1 = Registration2D3D_IntensityBased.MarginalProbabilityDensity(data1, numBins);
            mpd2 = Registration2D3D_IntensityBased.MarginalProbabilityDensity(data2, numBins);
            jpd = Registration2D3D_IntensityBased.JointProbability(data1, data2, numBins);
            pMI = log(jpd ./ (mpd1.*mpd2));
            if(isNormalized)
                pMI = pMI./(-log(jpd));
            end
        end

        function mpd = MarginalProbabilityDensity(data, numBins)
            data_norm = ceil(mat2gray(data)*numBins);       % change the range to [0 numBins]
            data_norm(data_norm==0) = 1;                    % [1 numBins]
            histogram = hist(data_norm(:), 1:numBins);      % intensity histogram
            pdf = histogram / sum(histogram(:));            % probability density function
            mpd = reshape( pdf(data_norm(:)), size(data) ); % move back to spatial image coordinate
        end

        function jpd = JointProbability(data1, data2, numBins)
            data1_norm = floor(mat2gray(data1)*numBins);  % change the range to [0 numBins]
            data1_norm(data1_norm==numBins) = numBins-1;  % [0 numBins-1]
            data2_norm = floor(mat2gray(data2)*numBins);  % change the range to [0 numBins]
            data2_norm(data2_norm==numBins) = numBins-1;  % [0 numBins-1]
            jointData = data2_norm * numBins + data1_norm + 1;  % range: [1 numBins*numBins]
            histogram = hist(jointData(:), 1:numBins*numBins);  % intensity joint histogram
            pdf = histogram / sum(histogram(:));            % probability density function
            jpd = reshape(pdf(jointData(:)), size(data1));  % move back to spatial image coordinate
        end 
        
        function [PDEs_mm, P_true_2D_mm, P_error_2D_mm] = ComputeProjectionDistanceError( true_9DOF, error_9DOF, targets, uv_dim, PixelSize )
            % computer projection distance error at specified target points
            % true and errored projection geometries are specified by 9DOF parameter
            %
            % \input 
            % true_9DOF (N x 9) : 9DOF projection parameter for the ground truth
            %                     col1-6: patient translation & rotation wrt detector 
            %                     col7-9: source position wrt detector
            % error_9DOF (N x 9) : 9DOF projection parameter for the error
            % targets (M x 3) : target points
            %
            % \output
            % PDEs (N x M) : Projection distance error (mm) at each projection
            % P_true_2D (N x M x 2): Projected target points at each true projection
            % P_error_2D (N x M x 2): Projected target points at each errored projection
            %

            M = size(targets, 1);
            PM_true = RegTools.generatePM_From9DOFParameters( true_9DOF, uv_dim, PixelSize );
            PM_error = RegTools.generatePM_From9DOFParameters( error_9DOF, uv_dim, PixelSize );

            p = reshape( permute(PM_true,[1 3 2]), [], 4) * [targets'; ones(1, M)];    % (3Nx4) x (4xM) = 3N x M
            P_true_2D_mm = cat(3, p(1:3:end,:)./p(3:3:end,:)*PixelSize(1), p(2:3:end,:)./p(3:3:end,:)*PixelSize(2)); % N x M x 2
            p = reshape( permute(PM_error,[1 3 2]), [], 4) * [targets'; ones(1, M)];    % (3Nx4) x (4xM) = 3N x M
            P_error_2D_mm = cat(3, p(1:3:end,:)./p(3:3:end,:)*PixelSize(1), p(2:3:end,:)./p(3:3:end,:)*PixelSize(2)); % N x M x 2

            PDEs_mm = sqrt(sum((P_true_2D_mm-P_error_2D_mm).^2,3));
        end
        
        function [samples, edge_lengths] = GenerateLocalSearchSpaces_KDTreePartitioning(range1xN, numPoints)
            dims = size(range1xN, 2);
            samples = zeros(numPoints, dims);
            edge_lengths = zeros(numPoints, dims); 
            volumes = zeros(numPoints, 1);
            dist_to_origin = zeros(numPoints, 1);

            % initialization
            edge_lengths(1,:) = range1xN*2;
            volumes(1) = prod(range1xN*2);
            dist_to_origin(1) = 0.0;

            for cur = 2:numPoints
                [largest_volume, largest_indx] = max( volumes(1:cur-1), [], 1 );
                % in case multiple cubes have the same volume, find all
                largests = find(volumes(1:cur-1)==largest_volume);
                % select the one closest to the origin
                [~, closest_to_origin] = sort(dist_to_origin(largests));
                tar = largests(closest_to_origin(1));    % this is the target box to split
            %     disp(['largest_volume = ' num2str(largest_volume') ', largest_indx = ' num2str(largest_indx')]);
            %     fprintf('target = %d, volume = %f\n', target, volumes(target));

                % we split the longest edge
                [~, sp_dim] = max( edge_lengths(tar,:), [], 2 );

                % split the target box (creating a new box)
                edge_lengths(cur,:) = edge_lengths(tar,:);
                edge_lengths([cur tar],sp_dim) = edge_lengths([cur tar],sp_dim)/2; % half edge length

                samples(cur,:) = samples(tar,:);
                samples([cur tar],sp_dim) = samples([cur tar],sp_dim) + [-1;1]*edge_lengths(tar,sp_dim)/2; % shift the center 1/2 of the total edge length

                volumes([tar cur]) = [1 1] * volumes(tar)/2;    % half volume
                dist_to_origin([tar cur]) = sqrt(sum(samples([tar cur],:).^2,2));   % recompute distance to the origin
            end

            % sort outputs
            [samples, indx] = sortrows(samples, 1:dims);
            edge_lengths = edge_lengths(indx,:);
        end
                
        function [out_Ubounds, out_Lbounds] = AdjustBounds(Ubounds_NxM, Lbounds_NxM, search_range_1xN, start_position_NxM)
            % adjust upper and lower bounds for multi-start
            num_multistart = size(start_position_NxM,2);
            ubounds_lim = (start_position_NxM+Ubounds_NxM)-search_range_1xN'*ones(1,num_multistart);
            ubounds_replace = (~isinf(ubounds_lim) & ubounds_lim>0);
            Ubounds_NxM(ubounds_replace>0) = Ubounds_NxM(ubounds_replace>0)-ubounds_lim(ubounds_replace);
            out_Ubounds = max(Ubounds_NxM, 0);

            lbounds_lim = (start_position_NxM+Lbounds_NxM)+search_range_1xN'*ones(1,num_multistart);
            lbounds_replace = (~isinf(lbounds_lim) & lbounds_lim<0);
            Lbounds_NxM(lbounds_replace>0) = Lbounds_NxM(lbounds_replace>0)-lbounds_lim(lbounds_replace);
            out_Lbounds = min(Lbounds_NxM, 0);
        end
    end
    
    methods (Access = protected)
        function SetVolume(obj, volume)
            if(~exist('volume','var')), volume = []; end
            obj.InputVolume = volume;
            if(iscell(obj.InputVolume))
                for i=1:length(obj.InputVolume)
                    obj.Param.VolumeDim(i,:) = size(obj.InputVolume{i});
                end
            else
                obj.Param.VolumeDim = size(volume);
            end
        end % SetVolume
        
        function SetModel(obj, model)
        end % SetModel
    end
end % Registration2D3DInterface