classdef Registration2D3D_IntensityBased_ObjectiveFunctionParameters < hgsetget
    % Parameters used in ObjectiveFunction for intensity-based 2D/3D registration
    % This class should be 'light' since we pass it around many times in
    % the optimization process
    properties (Access = public)
        InitialGuess = [];
        CurrentStartPose = [];
        NumImages;  % number of images

        % Step size (0: Siddon, -1: Auto)
        StepSize = -1;
                               
        % type of similarity measure. one of the following
        % see RegTools class for detail
        % SimilarityMeasureType_MI, SimilarityMeasureType_NMI,
        % SimilarityMeasureType_GI, SimilarityMeasureType_GC,
        % SimilarityMeasureType_NCC, SimilarityMeasureType_MSD,
        % SimilarityMeasureType_LogLikelihood
        SimilarityMeasureType;

        % parameters for LogLikelihood objective function
        I0 = 0;     % I0 value
        
        % registration log (6 parameters, cost and time for each iteration)
        % is returned to this cell array
        % RegistrationLogs: 1xn cell array (n: number of trials)
        % RegistrationLogs_LevelEnd: lxn matrix (l: number of resolution level)
        RegistrationLogs = cell(0);
        RegistrationLogs_LevelEnd = [];
        RecordLog = true;
        TimerID = -1;
        
        % followings are optional
        ShowProgressFunc = '';  % name of the function that shows progress
        ProgressShowStep = 200; % ShowProgressFunc is called every this iteration
        ProgressShowInterval_sec = Inf; % ShowProgressFunc is called every this interval (in sec)
        NextProgressShow = 0;
        NextProgressShowTime = 0;
        p_true = [];            % optional (for error analysis)
        Target3D = [];          % optional (for error analysis)
        Target2D = [];          % optional (for error analysis)
        TargetPointID;          % optional (for error analysis)
        TargetLabels;           % optional (for error analysis)
        TrialNo;                % optional (for evaluation with multiple trials)
        DatasetName;            % optional (for evaluation with multiple datasets)
        OriginalFixed;          % optional (fixed images in the original resolution. note that passing around large images slows down.)
        OriginalProjectionMatrices;
        
        % parameters that is used temporarily for optimizer
        similarity_measure_plan_id = int32([]);
        forwardProjection_plan_id = int32([]);
        current_iteration = 0; % number of current iteration
		current_generation = 0;
        display_result = false;
        record_file_id = 0;
        MaxParallel = inf;
		PopSize = 100;
		MaxGeneration = 100;
        
        num_multi_start = 1;
        current_start = 1;
        num_multi_level = 1;
        current_level = 1;
        initialSearchTime = 0;  % computation time for initial search (sec)
    
        PixelSize;  % in case where the objective function wants to use the original pixel size
        ImageDim;
        VoxelSize;  % in case where the objective function wants to use the original voxel size
        VolumeDim;
        CurrentDownsampleRatio_Projection;  % in case where the objective function wants to use the original pixel size
        CurrentDownsampleRatio_Volume;
        
        ROI_left_bottoms;
        ROI_square_size;
        
        % inclusion test for cmaes_ex.m
        cmaes_inclusionTest = '';
                
        % whether estimate source position or not
        isEstimateSourcePosition = false;
        
        % order of multiplication of the offset transformation matrix
        % Extrinsic parameter (Ext) is computed from the offset (Offset) and 
        % the current estimate (est) differently depending of the registration scenario
        % 'Post' multiplication: Ext = est * Offset
        %   (this is for scenarios where the camear is moving wrt the
        %   object, such as registration of endoscope camera image, thus we
        %   want to perturb T_camera_object matrix around the Offset)
        % 'Pre' multiplication: Ext = Offset * est 
        %   (this is for scenarios where the object is moving wrt the
        %   camera, such as registration of X-ray image, thus we want to
        %   perturb T_object_camera matrix around the Offset)
        OffsetMultiplicationOrder = 'Pre';   % default is the 'object moving' scenario
        
        % offset to the center of perturbation in the volume coordinate
        % this translation is added to the random perturbation from left
        % side (positive) and right side (negative), i.e.,
        % perturbation = T * perturbation * -T
        PerturbationCenter = [0 0 0];
		
		% penalty term weight
		Lambda_L2_penalty = 0;

		% alpha weight scheduling
		alpha_scheduling_mode = 0;
        
        % Parameters for the multiple bodies registration
        FixedVolumesIndices = [];
        MovingVolumesIndices = [];
    	FixedVolumesTransforms = [];
        FixedVolumesDRR = [];
        VolumeOffsets = [];

        % extra parameters for special case (e.g., piecewise-rigid
        % registration, etc.)
        extra_param;    % could be struct
    end
end
