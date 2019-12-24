% Author(s):  Yoshito Otake, Ali Uneri
% Created on: 2013-03-01

classdef RegTools < handle
    % MATLAB bindings of the GPU-accelerated Reconstruction Toolkit

    properties (Access = private)
        instance;
        ProjectionBufferPtr = -1;
        VolumeBufferPtr = -1;
        AllocatedProjectionBufferSize = [-1 -1 -1];
        AllocatedVolumeBufferSize = [-1 -1 -1 -1];
        SimilarityMeasureComputationTime;
    end
    properties (Constant = true)
        library = 'RegToolsLIB';
        headers = {'RegToolsMatlab.h', 'ProjectionParameterStructures.h'};
        proto = @RegToolsProto;
        protoName = 'RegToolsProto.m';
        
        MAX_MESSAGE_LENGTH = 2048;
        
        ProjectorMode_LinearInterpolation = 0;
        ProjectorMode_Siddon = 1;
        ProjectorMode_RayCasting = 5;
        ProjectorMode_LinearInterpolationDeformable = 6;

        MemoryStoreMode_Replace = 0;
        MemoryStoreMode_Additive = 1;
        MemoryStoreMode_Multiplicative = 2;
        
        SimilarityMeasureType_MI = 1;
        SimilarityMeasureType_NMI = 2;
        SimilarityMeasureType_GI = 3;
        SimilarityMeasureType_GI_SINGLE = 4;
        SimilarityMeasureType_GC = 5;
        SimilarityMeasureType_NCC = 6;
        SimilarityMeasureType_MSE = 7;
        SimilarityMeasureType_GI_SINGLE_ALWAYS_FLOATING_NORM = 9;
        SimilarityMeasureType_SSIM = 10;
        SimilarityMeasureType_GI_STD_NORM = 11;
        
        InterpolatorType_Bilinear = 0;
        InterpolatorType_Bicubic = 1;
        InterpolatorType_Bspline = 2;
        InterpolatorType_NearestNeighbor = 5;
    end

    methods
        function obj = RegTools(deviceIDs, GPUload_list, log_file_name)
            
            % need to do two checks here. In compiled code, deviceIDs may
            % be passed as an empty list
            if(~exist('deviceIDs', 'var') || isempty(deviceIDs))
                deviceIDs = -1; 
            end
            
            % Constructor.
            if not(libisloaded(obj.library))
                
                % try to load with proto file first
                if exist(obj.protoName, 'file') == 2
                    [notfound, warnings] = loadlibrary(obj.library, obj.proto);
                else
                    [notfound ,warnings] = loadlibrary(obj.library, obj.headers{1}, 'addheader', obj.headers{2});
                end
                
            end
            obj.instance = struct('InstancePtr', libpointer('voidPtr'));
            instancePtr = libpointer('RegToolsInstance', obj.instance);

            if ~calllib(obj.library, 'CreateRegToolsInstance', instancePtr)
                obj.check_error('CreateRegToolsInstance', true); 
            end
            obj.instance = instancePtr.value;
            
            if(exist('log_file_name','var') && ~isempty(log_file_name))
                if ~calllib(obj.library, 'AddLogFile', instancePtr, log_file_name)
                    obj.check_error('AddLogFile', true); 
                end
            end

            if(~exist('GPUload_list','var')), load_list_ptr = libpointer(); else load_list_ptr = libpointer('doublePtr', GPUload_list); end
            
            message_ptr = libpointer('stringPtrPtr', cellstr( repmat('0', 1, RegTools.MAX_MESSAGE_LENGTH) )); % dummy cell array of char
            if ~calllib(obj.library, 'InitializeRegToolsThread', instancePtr, libpointer('int32Ptr', deviceIDs), length(deviceIDs), load_list_ptr, message_ptr)
                obj.check_error('InitializeRegToolsThread', true); 
            else
                fprintf('%s', message_ptr.value{1}); % show messages
            end
            clear message_ptr;
            
            obj.SimilarityMeasureComputationTime = libpointer( 'singlePtr', 0 );
        end
        
        function delete(obj)
            % Destructor.
            if(obj.ProjectionBufferPtr ~= -1), obj.ProjectionBufferPtr = -1; end
            if(obj.VolumeBufferPtr ~= -1),     obj.VolumeBufferPtr = -1;     end
            clear obj.SimilarityMeasureComputationTime;
            message_ptr = libpointer('stringPtrPtr', cellstr( repmat('0', 1, RegTools.MAX_MESSAGE_LENGTH) )); % dummy cell array of char
            calllib(obj.library, 'DestroyRegToolsInstance', obj.instance, message_ptr);
            fprintf('%s', message_ptr.value{1}); % show messages
            clear message_ptr;
            unloadlibrary(obj.library);
        end

        function geometry_id = InitializeProjectionParametersArray(obj, numProjections)
            geometry_id = calllib(obj.library, 'InitializeProjectionParametersArray', obj.instance, numProjections);
        end

        function SetCurrentGeometrySetting(obj, geometry_id)
            calllib(obj.library, 'SetCurrentGeometrySetting', obj.instance, geometry_id);
        end

        function geometry_id = GetCurrentGeometrySetting(obj)
            geometry_id = calllib(obj.library, 'GetCurrentGeometrySetting', obj.instance);
        end

        function DeleteProjectionParametersArray(obj, geometry_id)
            calllib(obj.library, 'DeleteProjectionParametersArray', obj.instance, geometry_id);
        end

        function SetProjectionParameter_objectOriented(obj, projection_number, projectionParams)
            calllib(obj.library, 'SetProjectionParameter_objectOriented', obj.instance, projection_number, projectionParams);
        end
        
        function SetProjectionParameter_3x4ProjectionMatrix(obj, projection_number_0_base, pm, pixel_size, uv_dim, down_sample_ratio)
            % Be careful. This function seems to have a bug
            % the projection matrix 'pm' seems to be in row major, NOT
            % column major. When you set the matrix that was obtained by
            % "Get" function, you have to TRANSPOSE in 1st and 2nd dimension.
            if(~exist('down_sample_ratio', 'var')), down_sample_ratio = 1; end
            if(isscalar(pixel_size)), pixel_size = [pixel_size pixel_size]; end   % for backward compatibility (when only width is specified)
            if(isscalar(down_sample_ratio)), down_sample_ratio = [down_sample_ratio down_sample_ratio]; end   % for backward compatibility (when only u is specified)
            calllib(obj.library, 'SetProjectionParameter_3x4PM', obj.instance, projection_number_0_base, pm ...
                               , pixel_size(1), pixel_size(2), uv_dim(1), uv_dim(2), down_sample_ratio(1), down_sample_ratio(2));
        end

        function SetProjectionParameter_3x4ProjectionMatrix_Multi(obj, numProj, pm, pixel_size, uv_dim)
            % Be careful. This function seems to have a bug
            % the projection matrix 'pm' seems to be in row major, NOT
            % column major. When you set the matrix that was obtained by
            % "Get" function, you have to TRANSPOSE in 1st and 2nd dimension.
            if(isscalar(pixel_size)), pixel_size = [pixel_size pixel_size]; end   % for backward compatibility (when only width is specified)
            calllib(obj.library, 'SetProjectionParameter_3x4PM_multi', obj.instance, numProj, pm, pixel_size(1), pixel_size(2), uv_dim(1), uv_dim(2));
        end

        function SetVolumeTransform(obj, transform)
            if(length(transform)==6)
                % [tx ty tz rx ry rz]
                calllib(obj.library, 'SetWorldToVolumeTransform_6', obj.instance, transform(1), transform(2), transform(3), transform(4), transform(5), transform(6));
            elseif(size(transform)==[4 4])
                if(~strcmp(class(transform), 'double')), transform = double(transform); end
                calllib(obj.library, 'SetWorldToVolumeTransform_4x4', obj.instance, libpointer('doublePtr', transform));
            else
                disp('Error at SetVolumeTransform(), you should use either 4x4 matrix or 6 parameters');
            end
        end

        function SetVolumeInfo(obj, info)
            % info.VolumeDim: volume dimension
            % info.VoxelSize: voxel size
            if(length(info.VolumeDim)==1),  info.VolumeDim = [info.VolumeDim 1 1]; end;
            if(length(info.VolumeDim)==2),  info.VolumeDim = [info.VolumeDim 1]; end;
            calllib(obj.library, 'SetVolumeInfo', obj.instance, info.VolumeDim(1), info.VolumeDim(2), info.VolumeDim(3), ...
                                    info.VoxelSize(1), info.VoxelSize(2), info.VoxelSize(3));
        end

        function info = GetVolumeInfo(obj)
            % info.VolumeDim: volume dimension
            % info.VoxelSize: voxel size
            dim_x = libpointer( 'int32Ptr', 0 );        dim_y = libpointer( 'int32Ptr', 0 );        dim_z = libpointer( 'int32Ptr', 0 );
            size_x = libpointer( 'singlePtr', 0 );    size_y = libpointer( 'singlePtr', 0 );    size_z = libpointer( 'singlePtr', 0 );
            calllib(obj.library, 'GetVolumeInfo', obj.instance, dim_x, dim_y, dim_z, size_x, size_y, size_z);
            info.VolumeDim = [dim_x.value dim_y.value dim_z.value];
            info.VoxelSize = [size_x.value size_y.value size_z.value];
            clear dim_x; clear dim_y; clear dim_z; clear size_x; clear size_y; clear size_z;
        end

        function SetProjectionDim(obj, projectionImageSize )
            calllib(obj.library, 'SetProjectionDim', obj.instance, projectionImageSize(1), projectionImageSize(2));
        end

        function SetProjectorMode(obj, projectorMode ) % one of { ProjectorMode_Simple, ProjectorMode_Siddon, ProjectorMode_SF_TR, ProjectorMode_SF_TT, ProjectorMode_SF_DD }
            calllib(obj.library, 'SetProjectorMode', obj.instance, projectorMode);
        end

        function projector_mode = GetProjectorMode(obj)
            projector_mode = calllib(obj.library, 'GetProjectorMode', obj.instance);
        end

        function projection_dim = GetProjectionDim(obj)
            width_ptr = libpointer( 'int32Ptr', 0 );
            height_ptr = libpointer( 'int32Ptr', 0 );
            calllib(obj.library, 'GetProjectionDim', obj.instance, width_ptr, height_ptr);
            projection_dim = [width_ptr.value height_ptr.value];
            clear width_ptr;
            clear height_ptr;
        end

        function pm = GetProjectionMatrices(obj)
            num_projections = obj.GetNumberOfProjections();
            pm_ptr = libpointer( 'doublePtr', zeros(3, 4, num_projections) );
            calllib(obj.library, 'GetProjectionMatrices', obj.instance, pm_ptr);
            pm = permute(reshape(pm_ptr.value(:), [4 3 num_projections]), [2 1 3]); % row-major -> column-major
            clear pm_ptr;
        end

        function pixel_size = GetPixelSize(obj)
            width_ptr = libpointer( 'doublePtr', 0 );
            height_ptr = libpointer( 'doublePtr', 0 );
            calllib(obj.library, 'GetPixelSize', obj.instance, width_ptr, height_ptr);
            pixel_size = [width_ptr.value height_ptr.value];
            clear width_ptr;
            clear height_ptr;
        end

        function num_projections = GetNumberOfProjections(obj)
            num_projections = calllib(obj.library, 'GetNumberOfProjections', obj.instance);
        end

        function num_enabled_projections = GetNumberOfEnabledProjections(obj)
            num_enabled_projections = calllib(obj.library, 'GetNumberOfEnabledProjections', obj.instance);
        end

        function SetNumberOfProjectionSets(obj, num_projection_sets )
            calllib(obj.library, 'SetNumberOfProjectionSets', obj.instance, num_projection_sets);
        end

        function num_projection_sets = GetNumberOfProjectionSets(obj)
            num_projection_sets = calllib(obj.library, 'GetNumberOfProjectionSets', obj.instance);
        end
 
        function SetTransferBlockSize(obj, transfer_block_size )
            calllib(obj.library, 'SetTransferBlockSize', obj.instance, transfer_block_size);
        end
  
        function SetStepSize(obj, step_size )
            calllib(obj.library, 'SetStepSize', obj.instance, step_size);
        end

        function step_size = GetStepSize(obj)
            step_size_ptr = libpointer( 'singlePtr', 0 );
            calllib(obj.library, 'GetStepSize', obj.instance, step_size_ptr);
            step_size = step_size_ptr.value;
            clear step_size_ptr;
        end
  
        function SetRayCastingLOD(obj, lod )
            calllib(obj.library, 'SetRayCastingLOD', obj.instance, lod);
        end

        function lod = GetRayCastingLOD(obj)
            lod_ptr = libpointer( 'int32Ptr', 0 );
            calllib(obj.library, 'GetRayCastingLOD', obj.instance, lod_ptr);
            lod = lod_ptr.value;
            clear lod_ptr;
        end
  
        function SetRayCastingThreshold(obj, threshold )
            calllib(obj.library, 'SetRayCastingThreshold', obj.instance, threshold);
        end

        function threshold = GetRayCastingThreshold(obj)
            threshold_ptr = libpointer( 'singlePtr', 0 );
            calllib(obj.library, 'GetRayCastingThreshold', obj.instance, threshold_ptr);
            threshold = threshold_ptr.value;
            clear threshold_ptr;
        end
  
        function SetRayCastingDistanceFalloffCoefficient(obj, coeff )
            calllib(obj.library, 'SetRayCastingDistanceFalloffCoefficient', obj.instance, coeff);
        end

        function coeff = GetRayCastingDistanceFalloffCoefficient(obj)
            coeff_ptr = libpointer( 'singlePtr', 0 );
            calllib(obj.library, 'GetRayCastingDistanceFalloffCoefficient', obj.instance, coeff_ptr);
            coeff = coeff_ptr.value;
            clear coeff_ptr;
        end
  
        function SetCountNonIntersectedPixel(obj, count_non_intersected_pixel )
            calllib(obj.library, 'SetCountNonIntersectedPixel', obj.instance, count_non_intersected_pixel);
        end

        function count_non_intersected_pixel = GetCountNonIntersectedPixel(obj)
            count_non_intersected_pixel_ptr = libpointer( 'int32Ptr', 0 );
            calllib(obj.library, 'GetCountNonIntersectedPixel', obj.instance, count_non_intersected_pixel_ptr);
            count_non_intersected_pixel = count_non_intersected_pixel_ptr.value;
            clear count_non_intersected_pixel_ptr;
        end
  
        function SetDifferentVolumePerProjectionSet(obj, different_volume_per_projection_set )
            calllib(obj.library, 'SetDifferentVolumePerProjectionSet', obj.instance, different_volume_per_projection_set);
        end

        function different_volume_per_projection_set = GetDifferentVolumePerProjectionSet(obj)
            different_volume_per_projection_set_ptr = libpointer( 'int32Ptr', 0 );
            calllib(obj.library, 'GetDifferentVolumePerProjectionSet', obj.instance, different_volume_per_projection_set_ptr);
            different_volume_per_projection_set = different_volume_per_projection_set_ptr.value;
            clear different_volume_per_projection_set_ptr;
        end

        function SetVolumeToTexture(obj, volume)
            if(~strcmp(class(volume), 'single')), volume = single(volume); end
            calllib(obj.library, 'SetVolumeToTexture', obj.instance, libpointer('singlePtr', volume));
        end
        
        function SetSubSamplingVector_array(obj, sub_sampling_vector_array)
            % the vector represents which projection is enabled/disabled
            % 0: disabled, 1: enabled
            if(~strcmp(class(sub_sampling_vector_array), 'int32')), sub_sampling_vector = int32(sub_sampling_vector_array); end
            calllib(obj.library, 'SetSubSamplingVector', obj.instance, libpointer('int32Ptr', sub_sampling_vector_array), length(sub_sampling_vector_array));
        end
        
        function sub_sampling_vector = GetSubSamplingVector(obj)
            numProj = obj.GetNumberOfEnabledProjections();
            subSamplingVectorPtr = libpointer('int32Ptr', zeros(1,numProj,'int32'));
            calllib(obj.library, 'GetSubSamplingVector', obj.instance, subSamplingVectorPtr, numProj);
            sub_sampling_vector = subSamplingVectorPtr.value;
            clear subSamplingVectorPtr;
        end
        
        function SetSubSamplingVector_index(obj, sub_sampling_vector_indx)
            % the vector represents which projection is enabled/disabled
            if(~strcmp(class(sub_sampling_vector_indx), 'int32')), sub_sampling_vector_indx = int32(sub_sampling_vector_indx); end
            % if 'sub_sampling_vector' represents the indices of images
            subset = zeros(obj.GetNumberOfProjections(), 1); subset(sub_sampling_vector_indx) = 1; 
            sub_sampling_vector_array = subset;
            calllib(obj.library, 'SetSubSamplingVector', obj.instance, libpointer('int32Ptr', sub_sampling_vector_array), length(sub_sampling_vector_array));
        end
 
        function SetSubSamplingVector(obj, sub_sampling_vector_array)
            obj.SetSubSamplingVector_index(sub_sampling_vector_array);
        end
        
        function EraseDisabledProjections(obj)
            calllib(obj.library, 'EraseDisabledProjections', obj.instance);
        end
        
        function ReplicateProjections(obj, num_rep)
            calllib(obj.library, 'ReplicateProjections', obj.instance, num_rep);
        end
        
        function SetInitialProjectionOnDevice(obj, h_proj)
            h_proj = h_proj(:)';
            calllib(obj.library, 'SetInitialProjectionOnDevice', obj.instance, h_proj, numel(h_proj));
        end
        
        function ClearInitialProjectionOnDevice(obj)
            calllib(obj.library, 'ClearInitialProjectionOnDevice', obj.instance);
        end
        
        function geometry_id = GenerateGeometry_Simple(obj, geom)
            % Angles2 represent 'obliqueness' of the trajectory plane
            if(isempty(geom.Angles) && ~isempty(geom.angles)), geom.Angles = geom.angles; end
            if(~isfield(geom,'Angles2') || isempty(geom.Angles2)), geom.Angles2 = zeros(length(geom.Angles),1); end
            numProjections = length(geom.Angles);
            geometry_id = obj.InitializeProjectionParametersArray(numProjections);
            obj.SetProjectionDim(geom.UV_size);
            % origin of (u0, v0) is top-left corner of the image
            if(isstruct(geom))
                if(~isfield(geom, 'u0'))    geom.u0 = geom.UV_size(1)/2; end
                if(~isfield(geom, 'v0'))    geom.v0 = geom.UV_size(1)/2; end
                if(~isfield(geom, 'center_offset'))    geom.center_offset = [0 0 0]; end
            end
            if(length(geom.u0)==1), geom.u0 = repmat( geom.u0, [numProjections 1] ); end
            if(length(geom.v0)==1), geom.v0 = repmat( geom.v0, [numProjections 1] ); end
            % generate simple geometry (one-axis rotation trajectory)
            for i=1:numProjections
                u0 =   geom.UV_size(1)/2 - geom.u0(i);  % move the origin of u0 to center of the image
                v0 = -(geom.UV_size(2)/2 - geom.v0(i));  % move the origin of v0 to center of the image and flip the direction
                source_detector_base = obj.matRotationX(-90) * obj.matRotationY(90+geom.Angles(i)) * obj.matRotationX(geom.Angles2(i)) * obj.matTranslation(geom.center_offset(1),geom.center_offset(2),geom.center_offset(3));
                DetectorFrame = source_detector_base * obj.matTranslation(u0*geom.PixelSize(1), v0*geom.PixelSize(2), -(geom.SDD-geom.SAD));
                SourcePosition = source_detector_base * [0.0; 0.0; geom.SAD; 1.0];
                proj_struct = struct('DetectorFrame', DetectorFrame, 'SourcePosition', SourcePosition(1:3)', ...
                                     'FOV', geom.UV_size .* geom.PixelSize, 'Pixel_aspect_ratio', geom.PixelSize(1)/geom.PixelSize(2), 'Skew_angle_deg', 90, 'Pixel_width', geom.PixelSize(1));
                obj.SetProjectionParameter_objectOriented( i-1, proj_struct );  % note: 0-base index
            end
        end
        
        function geometry_id = GenerateGeometry_3x4ProjectionMatrix(obj, pm, pixel_size, uv_dim, down_sample_ratio)
            if(isempty(pm)), disp('Error in GenerateGeometry_3x4ProjectionMatrix()'); return; end
            numProjections = size(pm, 3);
            geometry_id = obj.InitializeProjectionParametersArray(numProjections);
            obj.SetProjectionDim(ceil(uv_dim));
            if(~exist('down_sample_ratio', 'var')), down_sample_ratio = 1; end
            if(isscalar(pixel_size)), pixel_size = [pixel_size pixel_size]; end   % for backward compatibility (when only width is specified)
            % generate geometry with projection matrices
            for i=1:numProjections
                obj.SetProjectionParameter_3x4ProjectionMatrix( i-1, libpointer( 'doublePtr', pm(:,:,i)' ), pixel_size, uv_dim, down_sample_ratio );  % note: 0-base index
            end
        end

        function [Bx, time] = ForwardProject_3x4ProjectionMatrices(obj, volume_id, pm_3x4 )
            if(nargout==0)
                calllib(obj.library, 'ForwardProjection_with3x4ProjectionMatrices', obj.instance, libstruct('ProjectionResult', struct('Data', libpointer('singlePtr'), ... 
                                       'projectionTime', libpointer('singlePtr'), 'minValue', libpointer, 'maxValue', libpointer, 'dDataID', -1, 'numGPU', -1, 'initData', libpointer('singlePtr') )), ...
                                       volume_id, libpointer('doublePtr', permute(pm_3x4,[2 1 3])) );
            else
                projection_buffer_size = [obj.GetProjectionDim() size(pm_3x4, 3)];
                if(any(obj.AllocatedProjectionBufferSize ~= projection_buffer_size))
                    obj.ProjectionBufferPtr = -1;   % delete existing memory space
%                         disp(['Allocate projection buffer in RegTools object: ' num2str(projection_buffer_size)]);
                    obj.ProjectionBufferPtr = obj.CreateProjectionImageBuffer( projection_buffer_size );
                    obj.AllocatedProjectionBufferSize = projection_buffer_size;
                end
                calllib(obj.library, 'ForwardProjection_with3x4ProjectionMatrices', obj.instance, obj.ProjectionBufferPtr, ...
                                        volume_id, libpointer('doublePtr', permute(pm_3x4,[2 1 3])));
                Bx = reshape( obj.ProjectionBufferPtr.Data, projection_buffer_size );
                time = obj.ProjectionBufferPtr.projectionTime;
            end
        end
        
        function [Bx, time] = ForwardProject(obj, volume, param1, param2, numView, computation_mode, numGPU, memory_store_mode, initData )
%         function [Bx, time] = ForwardProject(obj, volume, param1, param2, numView, numGPU, memory_store_mode ) % signature of previous version
            % param1:
            %   if volume is plan ID, transformations for each volume plan
            %   if volume is actual volume, max_value if necessary
            if(~exist('numGPU','var') || isempty(numGPU)), numGPU = -1; end
            if(~exist('memory_store_mode','var')), memory_store_mode = obj.MemoryStoreMode_Replace; end
            if(~exist('initData', 'var')), initData = []; end;
            
            projection_size = obj.GetProjectionDim();
            nb = obj.GetNumberOfEnabledProjections();

            if(strcmp(class(volume), 'int32'))
                % forward projection using pre-created 'plan'
                if(~exist('param1', 'var') || isempty(param1)), param1 = eye(4); end
                if(~exist('param2', 'var') || isempty(param2)), param2 = eye(4); end
                if(~exist('numView', 'var') || isempty(numView)), numView = nb; end
                if(~strcmp(class(param1), 'double')), param1 = double(param1); end
                if(~strcmp(class(param2), 'double')), param2 = double(param2); end
                if(~strcmp(class(numView), 'int32')), numView = int32(numView); end
                numTotalProj = numView*size(param1,3);
                if(numTotalProj ~= nb)
                    % set "SubSamplingArray" (enable only first numTotalProj projections)
                    sub_sampling = zeros(obj.GetNumberOfProjections(),1,'int32'); sub_sampling(1:numTotalProj)=1;
                    obj.SetSubSamplingVector_array(sub_sampling); nb = numTotalProj;
                end
                if(nargout==0)
                    if isempty(initData)
                        projectionResult = libstruct('ProjectionResult', struct('Data', libpointer('singlePtr'), ...
                                                     'projectionTime', libpointer('singlePtr'), 'minValue', libpointer, ...
                                                     'maxValue', libpointer, 'dDataID', -1, 'numGPU', numGPU, 'initData', libpointer('singlePtr') ));
                    else
                        initData = initData(:)';
                        projectionResult = libstruct('ProjectionResult', struct('Data', libpointer('singlePtr'), ...
                                                     'projectionTime', libpointer('singlePtr'), 'minValue', libpointer, ...
                                                     'maxValue', libpointer, 'dDataID', -1, 'numGPU', numGPU, 'initData', libpointer('singlePtr', initData) ));
                    end
                    calllib(obj.library, 'ForwardProjection_withPlan', obj.instance, projectionResult, volume, size(param1,3), ...
                                           libpointer('doublePtr', param1), numView, size(param2,3), libpointer('doublePtr', param2), memory_store_mode);
                else
                    projection_buffer_size = [projection_size nb];
                    if(any(obj.AllocatedProjectionBufferSize ~= projection_buffer_size))
                        obj.ProjectionBufferPtr = -1;   % delete existing memory space
%                         disp(['Allocate projection buffer in RegTools object: ' num2str(projection_buffer_size)]);
                        obj.ProjectionBufferPtr = obj.CreateProjectionImageBuffer( projection_buffer_size, initData);
                        obj.AllocatedProjectionBufferSize = projection_buffer_size;
                    end
                    obj.ProjectionBufferPtr.initData = initData;

                    calllib(obj.library, 'ForwardProjection_withPlan', obj.instance, obj.ProjectionBufferPtr, volume, size(param1,3), ...
                                           libpointer('doublePtr', param1), numView, size(param2,3), libpointer('doublePtr', param2), memory_store_mode);
                    Bx = reshape( obj.ProjectionBufferPtr.Data, projection_buffer_size );
                    time = obj.ProjectionBufferPtr.projectionTime;
                end
            else
%                 if(exist('param1', 'var') && ~isempty(param1)), obj.SetMaxValue( param1 ); end
                if(~strcmp(class(volume), 'single')), volume = single(volume); end
                if(nargout==0)
                    calllib(obj.library, 'ForwardProjection', obj.instance, libstruct('ProjectionResult', struct('Data', libpointer('singlePtr'), ... 
                                           'projectionTime', libpointer('singlePtr'), 'minValue', libpointer, 'maxValue', libpointer, 'dDataID', -1, 'numGPU', numGPU, 'initData', libpointer('singlePtr') )), libpointer('singlePtr', volume));
                else
                    projection_buffer_size = [projection_size nb];
                    if(any(obj.AllocatedProjectionBufferSize ~= projection_buffer_size))
                        obj.ProjectionBufferPtr = -1;   % delete existing memory space
%                         disp(['Allocate projection buffer in RegTools object: ' num2str(projection_buffer_size)]);
                        obj.ProjectionBufferPtr = obj.CreateProjectionImageBuffer( projection_buffer_size );
                        obj.AllocatedProjectionBufferSize = projection_buffer_size;
                    end
                    obj.ProjectionBufferPtr.numGPU = numGPU;
                    calllib(obj.library, 'ForwardProjection', obj.instance, obj.ProjectionBufferPtr, libpointer('singlePtr', volume));
                    obj.ProjectionBufferPtr.numGPU = -1;    % this member is used for only a special case of ForwardProjection at this point
                    Bx = reshape( obj.ProjectionBufferPtr.Data, projection_buffer_size );
                    time = obj.ProjectionBufferPtr.projectionTime;
                end
            end
        end
        
        function [Bx, time] = ForwardProject_d(obj, volume, varargin )
            [Bx_f, time] = obj.ForwardProject(volume, varargin{:});
            Bx = double(Bx_f);
        end
        
        function [interpolated, time] = Interpolation(obj, volumePlans, transforms, type, order, bicubic_a, back_ground, volume_center)
            if(~exist('back_ground','var')), back_ground = 0; end
            if(~exist('volume_center','var')), volume_center = []; end
            volumeInfo = obj.GetVolumeInfo();
            if(size(transforms,1)==4)
				num_transform_element = 12;		% 3 x 4 matrix (first 3 rows of a homogeneous matrix)
				num_transforms = size(transforms, 3);
				transforms = transforms(1:3,:,:);
			else
				num_transform_element = 6;
				num_transforms = size(transforms, 1);
				transforms = transforms';
			end
            
            projection_buffer_size = [volumeInfo.VolumeDim num_transforms];
            if(any(obj.AllocatedVolumeBufferSize ~= projection_buffer_size))
                obj.VolumeBufferPtr = -1;   % delete existing memory space
                obj.VolumeBufferPtr = obj.CreateProjectionImageBuffer( projection_buffer_size );
                obj.AllocatedVolumeBufferSize = projection_buffer_size;
            end
            if(~strcmp(class(transforms), 'single')), transforms = single(transforms); end
            
            % interpolation using pre-created 'plan'
            calllib(obj.library, 'Interpolation_withPlan', obj.instance, obj.VolumeBufferPtr, volumePlans, ...
                libpointer('singlePtr', transforms), num_transform_element, num_transforms, type, order, bicubic_a, back_ground, libpointer('singlePtr', volume_center) );

            interpolated = reshape( obj.VolumeBufferPtr.Data, projection_buffer_size );
            time = obj.VolumeBufferPtr.projectionTime;
        end
        
        function ComputeLinearCombination(obj, warp_device, def_mode_device, mode_weight)
            calllib(obj.library, 'ComputeLinearCombination', obj.instance, warp_device, def_mode_device, mode_weight);
        end
        
        function SetWarpTextures(obj, warps)
            calllib(obj.library, 'SetWarpTextures', obj.instance, libpointer('int32Ptr', warps));
        end
            
        function [warped, time] = ApplyDeformationField(obj, volumePlans, warps, type, order, bicubic_a, back_ground, ...
                volume_center, out_volumePlan, scattered_pnts_plan, transforms_4x4xN)
            if(~exist('back_ground','var') || isempty(back_ground)), back_ground = 0; end
            if(~exist('volume_center','var') || isempty(volume_center)), volume_center = []; end
            if(~exist('out_volumePlan','var') || isempty(out_volumePlan)), out_volumePlan = -1; end
            if(~exist('scattered_pnts_plan','var') || isempty(scattered_pnts_plan)), scattered_pnts_plan = -1; end
            volumeInfo = obj.GetVolumeInfo();
            
            if(nargout==0)
                projection_buffer_size = 0;
            elseif(scattered_pnts_plan>=0)
                projection_buffer_size = [obj.GetInterpolatorPlanVolumeInfo(scattered_pnts_plan) 1];
            else
                projection_buffer_size = [volumeInfo.VolumeDim 1];
            end
            if(any(obj.AllocatedVolumeBufferSize ~= projection_buffer_size))
                obj.VolumeBufferPtr = -1;   % delete existing memory space
                obj.VolumeBufferPtr = obj.CreateProjectionImageBuffer( projection_buffer_size );
                obj.AllocatedVolumeBufferSize = projection_buffer_size;
            end
            if(exist('out_volumePlan','var')), obj.VolumeBufferPtr.dDataID = out_volumePlan; end
            
            if(~exist('transforms_4x4xN','var') || isempty(transforms_4x4xN))
                transforms_3x4xN = [];
            else
                transforms_3x4xN = transforms_4x4xN(1:3,:,:);
            end
            % interpolation using pre-created 'plan'
            calllib(obj.library, 'ApplyDeformationField', obj.instance, obj.VolumeBufferPtr, volumePlans, ...
                libpointer('int32Ptr', warps), numel(warps), type, order, bicubic_a, back_ground, ...
                libpointer('singlePtr', volume_center), scattered_pnts_plan, ...
                libpointer('singlePtr', transforms_3x4xN), 12, size(transforms_3x4xN,3) );

            if(nargout > 0)
                warped = reshape( obj.VolumeBufferPtr.Data, projection_buffer_size );
            end
            time = obj.VolumeBufferPtr.projectionTime;
        end
        
        function GPU_volume_id = GPUMallocVolume(obj, volume)
            % use this function when voxel size is not used
            if(~strcmp(class(volume), 'single')), volume = single(volume); end
            volumeDim = [size(volume,1) size(volume,2) size(volume,3)];
            numVolumes = size(volume,4);
            GPU_volume_plan = libstruct('VolumePlan_cudaArray', ...
                                struct('h_volume', libpointer('singlePtr', volume), ... 
                                       'VolumeDim', libpointer('int32Ptr', volumeDim ), ...
                                       'VoxelSize', libpointer('doublePtr', [0 0 0] ), ...
                                       'numVolumes', numVolumes ...
                                   ));
            GPU_volume_id = int32(calllib(obj.library, 'CreateVolumePlan_cudaArray', obj.instance, GPU_volume_plan, 0 ));
            clear GPU_volume_plan;
        end
        
        function interpolator_plan_id = CreateInterpolatorPlan(obj, volume, voxel_size)
            if(~exist('voxel_size','var')), voxel_size = [1.0 1.0 1.0]; end
            if(~strcmp(class(volume), 'single')), volume = single(volume); end
            volumeDim = [size(volume,1) size(volume,2) size(volume,3)];
            numVolumes = size(volume,4);
            interpolator_plan = libstruct('VolumePlan_cudaArray', ...
                                struct('h_volume', libpointer('singlePtr', volume), ... 
                                       'VolumeDim', libpointer('int32Ptr', volumeDim ), ...
                                       'VoxelSize', libpointer('doublePtr', voxel_size ), ...
                                       'numVolumes', numVolumes ...
                                   ));
            interpolator_plan_id = int32(calllib(obj.library, 'CreateVolumePlan_cudaArray', obj.instance, interpolator_plan, 1 ));
            clear interpolator_plan;
        end
         
        function CopyDeviceMemoryToCudaArray(obj, cudaArrayID, deviceMemoryID, isCopyToAllDevices, volume_index_tex_1_base, volume_index_dev_1_base)
            % Note: this is a very slow operation because of internal reordering (e.g., space-filling curve) for cudaArray 
            % (see, for example, http://www.cudahandbook.com/uploads/Chapter_10._Texturing.pdf for detail)
            if(~exist('isCopyToAllDevices','var')),  isCopyToAllDevices = 1; end
            if(~exist('volume_index_tex_1_base','var')), volume_index_tex_1_base = 1; end
            if(~exist('volume_index_dev_1_base','var')), volume_index_dev_1_base = 1; end
            calllib(obj.library, 'CopyDeviceMemoryToCudaArray', obj.instance, cudaArrayID, deviceMemoryID, isCopyToAllDevices, volume_index_tex_1_base, volume_index_dev_1_base );
        end
         
        function CopyDeviceMemoryToCudaArray_Multi(obj, cudaArrayIDs, deviceMemoryID)
            % Note: this is a very slow operation because of internal reordering (e.g., space-filling curve) for cudaArray 
            % (see, for example, http://www.cudahandbook.com/uploads/Chapter_10._Texturing.pdf for detail)
            calllib(obj.library, 'CopyDeviceMemoryToCudaArray_Multi', obj.instance, libpointer('int32Ptr', cudaArrayIDs ), length(cudaArrayIDs), deviceMemoryID);
        end
       
        function forward_projection_plan_id = CreateForwardProjectionPlan(obj, volume, voxel_size)
            if(~strcmp(class(volume), 'single')), volume = single(volume); end
            volumeDim = [size(volume,1) size(volume,2) size(volume,3)];
            numVolumes = size(volume,4);
            forward_projection_plan = libstruct('VolumePlan_cudaArray', ...
                                struct('h_volume', libpointer('singlePtr', volume), ... 
                                       'VolumeDim', libpointer('int32Ptr', volumeDim ), ...
                                       'VoxelSize', libpointer('doublePtr', voxel_size ), ...
                                       'numVolumes', numVolumes ...
                                   ));
            forward_projection_plan_id = int32(calllib(obj.library, 'CreateVolumePlan_cudaArray', obj.instance, forward_projection_plan, 1 ));
            clear forward_projection_plan;
        end
        
        function [volumeDim voxelSize numVolumes] = GetInterpolatorPlanVolumeInfo(obj, interpolator_plan_id)
            volumeDimPtr = libpointer('int32Ptr', [0 0 0]);
            voxelSizePtr = libpointer('doublePtr', [0 0 0]);
            numVolumesPtr = libpointer('int32Ptr', 0);
            calllib(obj.library, 'GetVolumePlan_cudaArrayVolumeInfo', obj.instance, interpolator_plan_id, volumeDimPtr, voxelSizePtr, numVolumesPtr );
            volumeDim = volumeDimPtr.value;
            voxelSize = voxelSizePtr.value;
            numVolumes = numVolumesPtr.value;
            clear volumeDimPtr;
            clear voxelSizePtr;
            clear numVolumesPtr;
        end
        
        function [volume voxel_size] = GetInterpolatorPlan(obj, interpolator_plan_id)
            [volume voxel_size] = obj.GetGPUVolume(interpolator_plan_id, 1);
        end
        
        function projections = GetGPUProjection(obj, projection_set_1_base_index, numView)
            projection_buffer_size = [obj.GetProjectionDim() numView];
            if(any(obj.AllocatedProjectionBufferSize ~= projection_buffer_size))
                obj.ProjectionBufferPtr = -1;   % delete existing memory space
                obj.ProjectionBufferPtr = obj.CreateProjectionImageBuffer( projection_buffer_size );
                obj.AllocatedProjectionBufferSize = projection_buffer_size;
            end
            calllib(obj.library, 'GetGPUProjection', obj.instance, obj.ProjectionBufferPtr, projection_set_1_base_index);
            projections = reshape( obj.ProjectionBufferPtr.Data, projection_buffer_size );
        end
        
        function [GPU_volume voxel_size] = GetGPUVolume(obj, GPU_volume_id, isTextureMemory, volume_index_1_base, CPU_pointer)
            if(~exist('GPU_volume_id','var')), GPU_volume_id = -1; end
            if(~exist('isTextureMemory','var')), isTextureMemory = 0; end
            if(~exist('volume_index_1_base','var')), volume_index_1_base = 1; end
            if(GPU_volume_id<0)
                info = obj.GetVolumeInfo();
                volumeDim = info.VolumeDim; voxel_size = info.VoxelSize;
            else
                [volumeDim voxel_size] = obj.GetInterpolatorPlanVolumeInfo(GPU_volume_id);
            end
            if(volumeDim ~= [0 0 0])
                if(exist('CPU_pointer','var'))
                    calllib(obj.library, 'GetVolumePlan_cudaArrayVolume', obj.instance, GPU_volume_id, CPU_pointer, isTextureMemory, volume_index_1_base );
%                     GPU_volume = CPU_pointer.value;
                else    
                    volumePtr = libpointer('singlePtr', zeros(volumeDim, 'single'));
                    calllib(obj.library, 'GetVolumePlan_cudaArrayVolume', obj.instance, GPU_volume_id, volumePtr, isTextureMemory, volume_index_1_base );
                    GPU_volume = reshape( volumePtr.value, volumeDim );
                    clear volumePtr;
                end
            end
        end
        
        function SetGPUVolume(obj, GPU_volume_id, volume, isTextureMemory, volume_index)
            if(~exist('isTextureMemory','var')), isTextureMemory = 1; end
            if(~exist('volume_index','var')), volume_index = 1; end
            [volumeDim voxel_size] = obj.GetInterpolatorPlanVolumeInfo(GPU_volume_id);
            if(volumeDim ~= [size(volume,1),size(volume,2),size(volume,3)]), fprintf('SetGPUVolume error: volume size does not match\n'); return; end
            if(~strcmp(class(volume), 'single')), volume = single(volume); end
            calllib(obj.library, 'SetVolumePlan_cudaArrayVolume', obj.instance, GPU_volume_id, libpointer('singlePtr', volume), isTextureMemory, volume_index );
        end
        
        function MultGPUVolume(obj, GPU_volume_id, val)
            calllib(obj.library, 'MultVolumePlan', obj.instance, GPU_volume_id, single(val) );
        end
         
        function CMAES_popuation(obj, arz_id, arx_id, arxvalid_id, xmean_id, diagD_id, lbounds_id, ubounds_id)
            calllib(obj.library, 'CMAES_popuation', obj.instance, arz_id, arx_id, arxvalid_id, xmean_id, diagD_id, lbounds_id, ubounds_id );
        end
       
        function [similarity_measure_computation_plan_id, normalization_factor] = ...
                CreateSimilarityMeasureComputationPlan(obj, fixed_images, sigma, mask_weight, max_num_image_sets, GI_threshold, ...
                normalize_max_fixed, normalize_min_fixed, normalize_max_floating, normalize_min_floating, SSIM_DynamicRange, type)
            if(~strcmp(class(fixed_images), 'single')), fixed_images = single(fixed_images); end
            if(~exist('GI_threshold','var') || isempty(GI_threshold)), GI_threshold = -3.402823466e+38; end    % definition of -FLT_MAX from <float.h>
            if(~exist('normalize_max_fixed','var') || isempty(normalize_max_fixed)), normalize_max_fixed = 0; end
            if(~exist('normalize_min_fixed','var') || isempty(normalize_min_fixed)), normalize_min_fixed = 0; end
            if(~exist('normalize_max_floating','var') || isempty(normalize_max_floating)), normalize_max_floating = 0; end
            if(~exist('normalize_min_floating','var') || isempty(normalize_min_floating)), normalize_min_floating = 0; end
            if(~exist('SSIM_DynamicRange','var') || isempty(SSIM_DynamicRange)), SSIM_DynamicRange = 1.0; end
            if(~exist('type','var') || isempty(type)), type = RegTools.SimilarityMeasureType_GI_SINGLE; end
            if(~exist('mask_weight','var'))
                mask_weight = []; 
            elseif(~isempty(mask_weight))
                fixed_image_size = size(fixed_images); mask_size = size(mask_weight);
                if(any(size(fixed_image_size)~= size(mask_size)) || any(fixed_image_size~=mask_size))
                    disp('size of masks should be the same as fixed images');
                    similarity_measure_computation_plan_id = -1;
                    return;
                end
            end
            if(~exist('max_num_image_sets','var')), max_num_image_sets = 1; end
            imageDim = size(fixed_images);
            if(length(imageDim)==2), imageDim(3) = 1; end
            % sigma<0 means that we don't use gradient-based similarity measure
            SM_plan = libstruct('SimilarityMeasureComputationPlan', ...
                                struct('h_fixed_images', libpointer('singlePtr', fixed_images), ... 
                                       'ImageDim', libpointer('int32Ptr', imageDim ), ...
                                       'Sigma', single(sigma), ...
                                       'MaxNumImageSets', int32(max_num_image_sets), ...
                                       'NormalizeMax_fixed', single(normalize_max_fixed), ...
                                       'NormalizeMin_fixed', single(normalize_min_fixed), ...
                                       'NormalizeMax_floating', single(normalize_max_floating), ...
                                       'NormalizeMin_floating', single(normalize_min_floating), ...
                                       'h_mask_weight', libpointer('singlePtr', mask_weight), ...
                                       'SimilarityMeasureType', type, ... % for now, we assume single modality version is going to be used for GI
                                       'h_GI_threshold', GI_threshold, ...
                                       'h_SSIM_DynamicRange', SSIM_DynamicRange ...
                                   ));
            normalization_factorPtr = libpointer('doublePtr', 0);
            similarity_measure_computation_plan_id = int32(calllib(obj.library, 'CreateSimilarityMeasureComputationPlan', obj.instance, SM_plan, normalization_factorPtr ));
            normalization_factor = normalization_factorPtr.value;
            clear normalization_factorPtr;
            clear SM_plan;
        end
        
        function [imageDim, normalizationFactor] = GetSimilarityMeasureComputationPlanImageInfo(obj, similarity_measure_computation_plan_id, GPU_ID)
            if(~exist('GPU_ID','var')), GPU_ID = -1; end
            imageDimPtr = libpointer('int32Ptr', [0 0 0]);
            normalizationFactorPtr = libpointer('doublePtr', 0);
            calllib(obj.library, 'GetSimilarityMeasureComputationPlanImageInfo', obj.instance, similarity_measure_computation_plan_id, GPU_ID, imageDimPtr, normalizationFactorPtr );
            imageDim = imageDimPtr.value;
            normalizationFactor = normalizationFactorPtr.value;
            clear imageDimPtr;
            clear normalizationFactorPtr;
        end
        
        function [images] = GetSimilarityMeasureComputationPlan(obj, similarity_measure_computation_plan_id, image_type, GPU_ID, MI_bins, frame_no)
            if(~exist('GPU_ID','var') || isempty(GPU_ID)), GPU_ID = -1; end
            if(~exist('MI_bins','var') || isempty(MI_bins)), MI_bins = 256; end
            if(~exist('frame_no','var') || isempty(frame_no)), frame_no = 1; end    % 1-base index
            if(image_type == 9)
                % get joint histogram for MI, NMI computation
                imageDim = [MI_bins MI_bins 1];
            else
				imageDim = obj.GetSimilarityMeasureComputationPlanImageInfo(similarity_measure_computation_plan_id, GPU_ID);
            end
            if(imageDim ~= [0 0 0])
                imagesPtr = libpointer('singlePtr', zeros(imageDim, 'single'));
                calllib(obj.library, 'GetSimilarityMeasureComputationPlanImages', obj.instance ...
                        , similarity_measure_computation_plan_id, GPU_ID, imagesPtr, image_type, frame_no-1); % make frame_no to 0-base index
                images = reshape( imagesPtr.value, imageDim );
            end
            clear imagePtr;
        end
        
        function DeleteGPUVolume(obj, volume_plan_id)
            calllib(obj.library, 'DeleteVolumePlan_cudaArray', obj.instance, volume_plan_id );
        end
        
        function DeleteInterpolatorPlan(obj, interpolator_plan_id)
            calllib(obj.library, 'DeleteVolumePlan_cudaArray', obj.instance, interpolator_plan_id );
        end
        
        function DeleteForwardProjectionPlan(obj, forward_projection_plan_id)
            calllib(obj.library, 'DeleteVolumePlan_cudaArray', obj.instance, forward_projection_plan_id );
        end
        
        function DeleteSimilarityMeasureComputationPlan(obj, plan_id)
            calllib(obj.library, 'DeleteSimilarityMeasureComputationPlan', obj.instance, plan_id );
        end
        
        function [similarity_measure, time] = ComputeSimilarityMeasure(obj, plan_ids, similarity_type, numImageSet)
            similarity_measure_ptr = libpointer( 'doublePtr', zeros(numImageSet, 1) );

            if(length(plan_ids)==1)
                calllib(obj.library, 'ComputeSimilarityMeasure', obj.instance, plan_ids, similarity_type, numImageSet, similarity_measure_ptr, obj.SimilarityMeasureComputationTime );
            elseif(length(plan_ids)==2)
                calllib(obj.library, 'ComputeSimilarityMeasure2', obj.instance, plan_ids(1), plan_ids(2), similarity_type, numImageSet, similarity_measure_ptr, obj.SimilarityMeasureComputationTime );
            end
            similarity_measure = similarity_measure_ptr.value;
            time  = obj.SimilarityMeasureComputationTime.value;
            clear similarity_measure_ptr;
        end
        
        function [left_bottoms square_size in_out] = ComputeBoxProjectionBoundingSquare(obj, box_center, box_size, margin)
             % note: left_bottoms is '0-base' index
            nb = obj.GetNumberOfEnabledProjections();
            left_bottoms_ptr = libpointer('int32Ptr', zeros(nb*2, 1));
            square_size_ptr = libpointer('int32Ptr', zeros(2, 1));
            in_out_ptr = libpointer('int32Ptr', zeros(nb, 1));
            if(~strcmp(class(box_center), 'double')), box_center = double(box_center); end
            if(~strcmp(class(box_size), 'double')), box_size = double(box_size); end
            if(~exist('margin','var')), margin = 0; end
            calllib(obj.library, 'ComputeBoxProjectionBoundingSquare', obj.instance, left_bottoms_ptr, square_size_ptr, in_out_ptr, ...
                libpointer('doublePtr', box_center), libpointer('doublePtr', box_size), margin);
            left_bottoms = reshape( left_bottoms_ptr.value, [2 nb] )';
            square_size = square_size_ptr.value;
            in_out = in_out_ptr.value;
            clear left_bottoms_ptr;
            clear square_size_ptr;
            clear in_out_ptr;
        end
        
        function CropAllProjections(obj, left_bottoms, square_size)
            if(~strcmp(class(left_bottoms), 'int32')), left_bottoms = int32(left_bottoms); end
            if(~strcmp(class(square_size), 'int32')), square_size = int32(square_size); end
            calllib(obj.library, 'CropAllProjections', obj.instance, libpointer('int32Ptr', left_bottoms'), libpointer('int32Ptr', square_size));            
        end
       
        function [projection_out in_out] = CropProjections(obj, projection_in, box_center, box_size)
            [left_bottoms square_size in_out] = obj.ComputeBoxProjectionBoundingSquare(box_center, box_size);
            % generate 'sub-projection' geometry
            obj.CropAllProjections(left_bottoms, square_size); % left_bottoms:

            % crop projection images
            projection_out = zeros(square_size(1), square_size(2), size(projection_in, 3));
            for i=1:size(projection_in, 3), 
                projection_out(:,:,i) = projection_in(left_bottoms(i,1)+1:left_bottoms(i,1)+square_size(1), left_bottoms(i,2)+1:left_bottoms(i,2)+square_size(2), i); 
            end
        end
        
        function l_out = Downsample2DProjections(obj, DS_l, images, interpolator_type)
            % 2D down-sampling of multiple projection images
            %
            % DS_l: down-sample ratio for projection (doesn't need to be an integer)
            % images: nu x nv x (number of images)
            
            if(isempty(images)), l_out = []; return; end
            if(~exist('interpolator_type','var')), interpolator_type = RegTools.InterpolatorType_Bicubic; end
            if(DS_l==1), l_out = images; return; end
            projectionsDim = size( images );
            if(length(projectionsDim)==2), projectionsDim(3)=1; end
            l_out_dim = ceil([projectionsDim(1) projectionsDim(2)]./DS_l);
            out_voxel_size = projectionsDim(1:2) ./ l_out_dim(1:2); % we don't want to change the FOV
            l_out = zeros( [l_out_dim projectionsDim(3)], 'single' );
            info = obj.GetVolumeInfo(); % store current volume size and voxel size
            for i=1:projectionsDim(3)
                projectionPlan = obj.CreateInterpolatorPlan( images(:,:,i), [1 1 1] );
                obj.SetVolumeInfo( struct('VolumeDim', [l_out_dim 1], 'VoxelSize', [out_voxel_size 1]) );
                l_out(:,:,i) = obj.Interpolation( projectionPlan, [0 0 0 0 0 0], interpolator_type, 0, -0.5 );
                obj.DeleteInterpolatorPlan( projectionPlan );
            end
            obj.SetVolumeInfo( info );
        end        
        
        function out_volume = Downsample3DVolumes(obj, DS, volumes, interpolator_type)
            % 3D down-sampling of multiple volumes
            %
            % DS: down-sample ratio for projection (doesn't need to be an integer)
            % volumes: nx x ny x nz x (number of volumes)
            
            if(isempty(volumes)), out_volume = []; return; end
            if(~exist('interpolator_type','var')), interpolator_type = RegTools.InterpolatorType_Bicubic; end
            if(DS==1), out_volume = volumes; return; end
            volumeDim = size( volumes );
            if(length(volumeDim)==3), volumeDim(4)=1; end
            out_dim = ceil([volumeDim(1) volumeDim(2) volumeDim(3)]./DS); % so that output volume dimension becomes integer
            out_voxel_size = volumeDim(1:3) ./ out_dim(1:3);
            out_volume = zeros( [out_dim volumeDim(4)], 'single' );
            info = obj.GetVolumeInfo(); % store current volume size and voxel size
            for i=1:volumeDim(4)
                volumePlan = obj.CreateInterpolatorPlan( volumes(:,:,:,i), [1 1 1] );
                obj.SetVolumeInfo( struct('VolumeDim', out_dim, 'VoxelSize', out_voxel_size) );
                out_volume(:,:,:,i) = obj.Interpolation( volumePlan, [0 0 0 0 0 0], interpolator_type, 0, -0.5 );
                obj.DeleteInterpolatorPlan( volumePlan );
            end
            obj.SetVolumeInfo( info );
        end        
                
        function [w12X, w12Y, w12Z] = ComposeDeformationField(obj, w1X, w1Y, w1Z, w2X, w2Y, w2Z)
            if(nargin == 3)
                % when the warp is specified in a format of [nx, ny, nz, 3] matrix
                w2Z = w1Y(:,:,:,3); w2Y = w1Y(:,:,:,2); w2X = w1Y(:,:,:,1); 
                w1Z = w1X(:,:,:,3); w1Y = w1X(:,:,:,2); w1X = w1X(:,:,:,1); 
            end
            w1_IDs = zeros(3, 1, 'int32'); w2_IDs = zeros(3, 1, 'int32');
            w1_IDs(1) = obj.CreateInterpolatorPlan(w1X); w1_IDs(2) = obj.CreateInterpolatorPlan(w1Y); w1_IDs(3) = obj.CreateInterpolatorPlan(w1Z);
            w2_IDs(1) = obj.CreateInterpolatorPlan(w2X); w2_IDs(2) = obj.CreateInterpolatorPlan(w2Y); w2_IDs(3) = obj.CreateInterpolatorPlan(w2Z);

            w12X = w2X + obj.ApplyDeformationField( w1_IDs(1), w2_IDs, RegTools.InterpolatorType_Bilinear, 0, -0.5, 0 );
            w12Y = w2Y + obj.ApplyDeformationField( w1_IDs(2), w2_IDs, RegTools.InterpolatorType_Bilinear, 0, -0.5, 0 );
            w12Z = w2Z + obj.ApplyDeformationField( w1_IDs(3), w2_IDs, RegTools.InterpolatorType_Bilinear, 0, -0.5, 0 );
            
            for i=1:3, obj.DeleteInterpolatorPlan(w1_IDs(i)); obj.DeleteInterpolatorPlan(w2_IDs(i)); end
            if(nargout == 1)
                % return [nx, ny, nz, 3] matrix
                w12X = cat(4, w12X, w12Y, w12Z);
            end
        end
        
        function v = InvertWarp(obj, warp, num_iterations, interpolator_type, v_GPU_tex, v_GPU_device)
            % compute inverse of deformation field using a simple iterative
            % approach (see Chen et al., MedPhys 35(1), pp.81-88, 2008)
            if(~exist('num_iterations','var') || isempty(num_iterations)), num_iterations = 10; end
            if(~exist('interpolator_type','var') || isempty(interpolator_type)), interpolator_type = RegTools.InterpolatorType_Bicubic; end
            
            if(isa(warp,'int32'))
                u_GPU = warp;
                volumeDim = obj.GetInterpolatorPlanVolumeInfo(warp(1));
                v = zeros([volumeDim numel(warp)],'single');
            else
                u_GPU = zeros(3, 1, 'int32');
                for d=1:3, u_GPU(d) = obj.CreateInterpolatorPlan(warp(:,:,:,d)); end
                v = zeros(size(warp),'single');
            end
            
            % allocate temporary GPU memory
            if(~exist('v_GPU_tex','var'))
                v_GPU_tex = zeros(3, 1, 'int32');
                for d=1:3, v_GPU_tex(d) = obj.CreateInterpolatorPlan(v(:,:,:,d)); end
                v_GPU_tex_allocated = true;
            else
                % just initialize v_GPU_tex (this is very slow because it
                % involves z-ordering)
                for d=1:3, obj.SetGPUVolume(v_GPU_tex(d), v(:,:,:,d), 1); end
                v_GPU_tex_allocated = false;
            end
            if(~exist('v_GPU_device','var'))
                v_GPU_device = zeros(3, 1, 'int32');
                for d=1:3, v_GPU_device(d) = obj.GPUMallocVolume(v(:,:,:,d)); end
                v_GPU_device_allocated = true;
            else
                v_GPU_device_allocated = false;
            end
            
            % iterative update, v_n(x) = -u(x + v_n-1(x))
            % we can check convergence by checking changes of v (e.g.,
            % median of magnitude of v vectors), but this slows down a lot
            % (we need to check this on the GPU)
%             v_prev = v; v_cur = v;
            for iteration=1:num_iterations
                for d=1:3
                    % output result to writable memory
                    obj.ApplyDeformationField( u_GPU(d), v_GPU_tex, interpolator_type, 0, -0.5, 0, [], v_GPU_device(d));
                    obj.MultGPUVolume( v_GPU_device(d), -1 );
                end
                for d=1:3, obj.CopyDeviceMemoryToCudaArray( v_GPU_tex(d), v_GPU_device(d), 1 ); end % copy to texture memory
%                 for d=1:3, v_cur(:,:,:,d) = obj.GetGPUVolume(v_GPU(d),1); end
%                 v_diff = v_prev-v_cur;  v_prev = v_cur;
%                 fprintf('iteration: %d, median(||v_diff||2) = %f\n', iteration, median(reshape(sqrt(sum(v_diff.^2,4)),[],1)) );
            end
            if(nargout>0)
                for d=1:3, v(:,:,:,d) = obj.GetGPUVolume(v_GPU_device(d)); end
            end
            
            % release GPU memory
            if(~isa(warp,'int32')), for d=1:3, obj.DeleteInterpolatorPlan(u_GPU(d)); end; end
            if(v_GPU_tex_allocated), for i=1:3, obj.DeleteInterpolatorPlan(v_GPU_tex(i)); end; end
            if(v_GPU_device_allocated), for i=1:3, obj.DeleteInterpolatorPlan(v_GPU_device(i)); end; end
        end
        
        function freeMem_byte = GPUmemCheck( obj, message, threadID )
            if(~exist('message','var')), message = 'GPUmemCheck'; end
            if(~exist('threadID','var')), threadID = 0; end
            freeMem_byte = calllib(obj.library, 'GPUmemCheck', obj.instance, libpointer('cstring', message), threadID);
        end
        
        function check_error(obj, callerFunction, destroyObj)
            % CHECKERROR - generates an error string and throws a
            % LibraryCall error
            % 
            %   obj.check_error(callerFuction[,destroyObj]);
            
            if nargin >=3 && destroyObj, delete(obj); end
            error( 'error at %s', callerFunction );
        end % check_error 
        
        function matrix_4x4 = convertTransRotTo4x4_C( obj, p )
            ret_ptr = libpointer( 'doublePtr', zeros(4, 4) );
            calllib(obj.library, 'convertTransRotTo4x4', obj.instance, libpointer('doublePtr', p), ret_ptr);
            matrix_4x4 = ret_ptr.value;
            clear ret_ptr;
        end
        
        function matrix_4x4 = convertRotTransTo4x4_C( obj, p )
            ret_ptr = libpointer( 'doublePtr', zeros(4, 4) );
            calllib(obj.library, 'convertRotTransTo4x4', obj.instance, libpointer('doublePtr', p), ret_ptr);
            matrix_4x4 = ret_ptr.value;
            clear ret_ptr;
        end
        
        function vec6 = convert4x4ToTransRot_C( obj, mat4x4 )
            ret_ptr = libpointer( 'doublePtr', zeros(1, 6) );
            calllib(obj.library, 'convert4x4ToTransRot', obj.instance, libpointer('doublePtr', mat4x4), ret_ptr);
            vec6 = ret_ptr.value;
            clear ret_ptr;
        end
        
        function vec6 = convert4x4ToRotTrans_C( obj, mat4x4 )
            ret_ptr = libpointer( 'doublePtr', zeros(1, 6) );
            calllib(obj.library, 'convert4x4ToRotTrans', obj.instance, libpointer('doublePtr', mat4x4), ret_ptr);
            vec6 = ret_ptr.value;
            clear ret_ptr;
        end
    end
   
    methods(Static)
        function list = GetGPUList()
            max_list_length = 20;
            max_name_length = 512;
            ptr = libpointer('stringPtrPtr', cellstr( repmat('0', max_list_length, max_name_length) )); % dummy cell array of char
            if not(libisloaded(RegTools.library))
                if exist(RegTools.protoName,'file') == 2
                    [notfound, warnings] = loadlibrary(RegTools.library, RegTools.proto);
                else
                    [notfound ,warnings] = loadlibrary(RegTools.library, RegTools.headers{1}, 'addheader', RegTools.headers{2});
                end
                numGPU = calllib(RegTools.library, 'GetGPUList', ptr, max_list_length, max_name_length);
                unloadlibrary(RegTools.library);
            else
                numGPU = calllib(RegTools.library, 'GetGPUList', ptr, max_list_length, max_name_length);
            end
            list = ptr.value(1:min(numGPU,max_list_length));
            clear ptr;
            for i=1:length(list), fprintf('%s\n', list{i}); end
        end
        
        function imageBufferPtr = CreateProjectionImageBuffer(dimensions, initData)
            arraySize = prod(double(dimensions));
            if(~exist('initData', 'var'))
                initDataPointer = libpointer('singlePtr');
            else
                initDataPointer = libpointer('singlePtr', initData);
            end;
                                     
            imageBufferPtr = libstruct('ProjectionResult', ...
                                struct('Data', libpointer('singlePtr', zeros([arraySize, 1], 'single')), ... 
                                       'projectionTime', libpointer('singlePtr', single(0.0) ), 'minValue', libpointer, 'maxValue', libpointer, 'dDataID', -1, 'numGPU', -1, 'initData', initDataPointer));
        end
        
        function pm = ComputeProjectionMatrices_Simple(geom)
            numProjections = length(geom.Angles);
            % origin of (u0, v0) is top-left corner of the image
            if(~isfield(geom, 'u0'))    geom.u0 = geom.UV_size(1)/2; end
            if(~isfield(geom, 'v0'))    geom.v0 = geom.UV_size(2)/2; end
            if(length(geom.u0)==1), geom.u0 = repmat( geom.u0, numProjections ); end
            if(length(geom.v0)==1), geom.v0 = repmat( geom.v0, numProjections ); end
            if(~isfield(geom, 'center_offset'))    geom.center_offset = [0 0 0]; end
            % generate simple geometry (one-axis rotation trajectory)
            pm = zeros(3, 4, numProjections);
            for i=1:numProjections
                u0 =   geom.UV_size(1)/2 - geom.u0(i);  % move the origin of u0 to center of the image
                v0 = -(geom.UV_size(2)/2 - geom.v0(i));  % move the origin of v0 to center of the image and flip the direction
                source_detector_base = RegTools.matRotationX(-90) * RegTools.matRotationY(90+geom.Angles(i)) * RegTools.matTranslation(geom.center_offset(1),geom.center_offset(2),geom.center_offset(3));
                DetectorFrame = source_detector_base * RegTools.matTranslation(u0*geom.PixelSize(1), v0*geom.PixelSize(2), -(geom.SDD-geom.SAD));
                SourcePosition = source_detector_base * [0.0; 0.0; geom.SAD; 1.0];
                extrinsic = inv([DetectorFrame(1:3,1:3), SourcePosition(1:3); [0 0 0 1]]);
                
                % compute detector position in source frame (sFw * d)
                D_on_S = extrinsic*[DetectorFrame(1:3,4); 1];
                intrinsic = [ [D_on_S(3)/geom.PixelSize(1); 0; 0] [0; D_on_S(3)/geom.PixelSize(2); 0] [geom.u0(i); geom.v0(i)-1; 1] ];
                pm(:,:,i) = [intrinsic [0;0;0]] * extrinsic;
                
                % scale projection matrix so that z=1 at the plane where width of one pixel becomes pixel_width 
                % (for distance weight in linear interpolation back projector)
                inv_pm3x3 = inv(pm(1:3,1:3,i));
                pm(:,:,i) = pm(:,:,i) * ( norm(inv_pm3x3(1:3,1)) / geom.PixelSize(1) );
            end
        end
        
        function PMs = ShiftProjectionMatrices(PMs, xy_shift)
            % shift origin of 3x4 projection matrix
            %   PMs: 3 x 4 x n matrix (n: number of views)
            %   xy_shift: n x 2 matrix (n: number of views), amount of shift for each view
            for i=1:size(PMs,3)
                % apply viewport transform multiply 
                % [1 0 -x_shift; 0 1 -y_shift; 0 0 1] from left side
                PMs(1,:,i) = PMs(1,:,i)-xy_shift(i,1)*PMs(3,:,i);
                PMs(2,:,i) = PMs(2,:,i)-xy_shift(i,2)*PMs(3,:,i);
            end
        end
 
        % utility functions
        function matrix_4x4 = convertTransRotTo4x4( p )
            % convert 6 parameter ([tx, ty, tz, rx, ry, rz]) to 4x4 transformation matrix
            % 6 parameter is represented in the order of (x,y,z) of translation and rotation
            % translation precede rotation
            % rotation angle is represented in unit of 'degree'
%             matrix_4x4 = RegTools.matTranslation( p(1), p(2), p(3) ) * RegTools.matRotationZ(p(6)) * RegTools.matRotationY(p(5)) * RegTools.matRotationX(p(4));
            angle_rad = pi/180.0 * p(4:6);
            c = cos(angle_rad); s = sin(angle_rad);
            matrix_4x4 = [c(2)*c(3) -c(1)*s(3)+s(1)*s(2)*c(3) s(1)*s(3)+c(1)*s(2)*c(3)  p(1);
                          c(2)*s(3) c(1)*c(3)+s(1)*s(2)*s(3)  -s(1)*c(3)+c(1)*s(2)*s(3) p(2);
                            -s(2)         s(1)*c(2)                   c(1)*c(2)         p(3);
                            0                   0                       0               1 ];
        end
        
        function matrix_4x4 = convertTransRotTo4x4_rad( p )
            % convert 6 parameter ([tx, ty, tz, rx, ry, rz]) to 4x4 transformation matrix
            % 6 parameter is represented in the order of (x,y,z) of translation and rotation
            % translation precede rotation
            % rotation angle is represented in unit of 'radians'
            c = cos(p(4:6)); s = sin(p(4:6));
            matrix_4x4 = [c(2)*c(3) -c(1)*s(3)+s(1)*s(2)*c(3) s(1)*s(3)+c(1)*s(2)*c(3)  p(1);
                          c(2)*s(3) c(1)*c(3)+s(1)*s(2)*s(3)  -s(1)*c(3)+c(1)*s(2)*s(3) p(2);
                            -s(2)         s(1)*c(2)                   c(1)*c(2)         p(3);
                            0                   0                       0               1 ];
        end
        
        function matrix_4x4xN = convertTransRotTo4x4_multi(p_6xN, offset_mat, offset_side)
            if(~exist('offset_side','var')), offset_side = 'Post'; end
            % convert 6 parameter ([tx, ty, tz, rx, ry, rz]) to 4x4 transformation matrix
            % multiple conversions are vectorized
            if(isvector(p_6xN)), p_6xN = p_6xN(:); end
            N = size(p_6xN,2);
            c = cos(pi/180.0*p_6xN(4:6,:)); s = sin(pi/180.0*p_6xN(4:6,:));
            zeros_1xN = zeros(1,N);
            if(~exist('offset_mat','var'))
                matrix_4x4xN = reshape( ...
                [ c(2,:).*c(3,:);                         c(2,:).*s(3,:); -s(2,:); zeros_1xN;
                  -c(1,:).*s(3,:)+s(1,:).*s(2,:).*c(3,:); c(1,:).*c(3,:)+s(1,:).*s(2,:).*s(3,:); s(1,:).*c(2,:); zeros_1xN;
                  s(1,:).*s(3,:)+c(1,:).*s(2,:).*c(3,:); -s(1,:).*c(3,:)+c(1,:).*s(2,:).*s(3,:); c(1,:).*c(2,:); zeros_1xN;
                  p_6xN(1,:);                            p_6xN(2,:);                            p_6xN(3,:);      ones(1,N)], 4,4,N);
            elseif(strcmp(offset_side, 'Pre'))
                % multiply offset from left-hand side
                [M, ~] = size(offset_mat); % offset_mat could be any matrix with Mx4
                matrix_4x4N = reshape( ...
                    [ c(2,:).*c(3,:);                         c(2,:).*s(3,:); -s(2,:); zeros_1xN;
                      -c(1,:).*s(3,:)+s(1,:).*s(2,:).*c(3,:); c(1,:).*c(3,:)+s(1,:).*s(2,:).*s(3,:); s(1,:).*c(2,:); zeros_1xN;
                      s(1,:).*s(3,:)+c(1,:).*s(2,:).*c(3,:); -s(1,:).*c(3,:)+c(1,:).*s(2,:).*s(3,:); c(1,:).*c(2,:); zeros_1xN;
                      p_6xN(1,:);                            p_6xN(2,:);                            p_6xN(3,:);      ones(1,N)], 4,4*N);
                matrix_4x4xN = reshape( offset_mat*matrix_4x4N, M,4,N );
            elseif(strcmp(offset_side, 'Post'))
                % multiply offset from right-hand side
                % note we use the relationship: A'*B' = (BA)'
                [~, M] = size(offset_mat); % offset_mat could be any matrix with 4xM
                matrix_4x4N = reshape( ...
                    [ c(2,:).*c(3,:); -c(1,:).*s(3,:)+s(1,:).*s(2,:).*c(3,:); s(1,:).*s(3,:)+c(1,:).*s(2,:).*c(3,:); p_6xN(1,:); ...
                      c(2,:).*s(3,:); c(1,:).*c(3,:)+s(1,:).*s(2,:).*s(3,:); -s(1,:).*c(3,:)+c(1,:).*s(2,:).*s(3,:); p_6xN(2,:); ...
                      -s(2,:);        s(1,:).*c(2,:);                         c(1,:).*c(2,:);                        p_6xN(3,:); ...
                      zeros_1xN;      zeros_1xN;                              zeros_1xN;                             ones(1,N)], 4,4*N);
                    
                matrix_4x4xN = permute( reshape( offset_mat'*matrix_4x4N, M,4,N ), [2 1 3] );
            end
        end
        
        function matrix_4x4 = convertTransRotZYXTo4x4( p )
            % convert 6 parameter ([tx, ty, tz, rx, ry, rz]) to 4x4 transformation matrix
            % 6 parameter is represented in the order of (x,y,z) of translation and rotation
            % translation precede rotation
            % rotation angle is represented in unit of 'degree' (Z->Y->X)
            matrix_4x4 = RegTools.matTranslation( p(1), p(2), p(3) ) * RegTools.matRotationX(p(4)) * RegTools.matRotationY(p(5)) * RegTools.matRotationZ(p(6));
        end
        
        function matrix_4x4 = convertRotTransTo4x4( p )
            % convert 6 parameter ([tx, ty, tz, rx, ry, rz]) to 4x4 transformation matrix
            % rotation precede translation
            % rotation angle is represented in unit of 'degree'
            matrix_4x4 = RegTools.matRotationZ(p(6)) * RegTools.matRotationY(p(5)) * RegTools.matRotationX(p(4)) * RegTools.matTranslation( p(1), p(2), p(3) );
        end
        
        function vec6 = convert4x4ToTransRot( mat4x4 )
            % convert 4x4 transformation matrix to 6 parameter ([tx, ty, tz, rx, ry, rz])
            % translation precede rotation (used in Projector)
            % rotation is represented X->Y->Z Euler angle in unit of 'degree'
            vec6 = [mat4x4(1:3,4)' RegTools.RotMat2RPY(mat4x4(1:3,1:3))'];
        end
        
        function vec6xN = convert4x4ToTransRot_multi( mat4x4 )
            % convert 4x4 transformation matrix to 6 parameter ([tx, ty, tz, rx, ry, rz])
            % translation precede rotation (used in Projector)
            % rotation is represented X->Y->Z Euler angle in unit of 'degree'
            vec6xN = [permute(mat4x4(1:3,4,:),[3 1 2])'; RegTools.RotMat2RPY(mat4x4(1:3,1:3,:))];
        end
        
        function vec6 = convert4x4ToTransRotZYX( mat4x4 )
            % convert 4x4 transformation matrix to 6 parameter ([tx, ty, tz, rx, ry, rz])
            % translation precede rotation (used in Projector)
            % rotation is represented Z->Y->X Euler angle in unit of 'degree'
            vec6 = [mat4x4(1:3,4)' RegTools.Rot2XYZ_Euler(mat4x4(1:3,1:3))'];
        end
        
        function vec6 = convert4x4ToRotTrans( mat4x4 )
            % convert 4x4 transformation matrix to 6 parameter ([tx, ty, tz, rx, ry, rz])
            % rotation precede translation (used in Interpolator)
            % rotation is represented X->Y->Z Euler angle in unit of 'degree'
            inv_mat4x4 = inv(mat4x4);
            vec6 = [-inv_mat4x4(1:3,4)' -RegTools.Rot2XYZ_Euler(inv_mat4x4(1:3,1:3))'];
        end
        
        function vec6 = convert4x4ToRotTrans_rad( mat4x4 )
            % convert 4x4 transformation matrix to 6 parameter ([tx, ty, tz, rx, ry, rz])
            % rotation precede translation (used in Interpolator)
            % rotation is represented X->Y->Z Euler angle in unit of 'radian'
            vec6 = RegTools.convert4x4ToRotTrans( mat4x4 ) .* [1 1 1 pi/180 pi/180 pi/180];
        end
        
        function xyz = RotMat2RPY(R)
            % compute Roll-Pitch-Yaw angle (X->Y->Z Euler angle) from a
            % rotation matrix.
            % return column vector in unit of 'degree'
            R = reshape(R, 9, []);
            pitch = atan2(-R(3,:), sqrt(R(1,:).^2 + R(2,:).^2));
            if abs(pitch - pi/2) < eps
              yaw = zeros(1,size(R,2));  roll = atan2(R(4,:), R(5,:));
            elseif abs(pitch + pi/2) < eps
              yaw = zeros(1,size(R,2));  roll = -atan2(R(4,:), R(5,:));
            else
              yaw = atan2(R(2,:)./cos(pitch), R(1,:)./cos(pitch));
              roll = atan2(R(6,:)./cos(pitch), R(9,:)./cos(pitch));
            end
            xyz = [roll; pitch; yaw].*180/pi;
        end
        
        function XYZ_Euler = Rot2XYZ_Euler(R)
        %returns the X-Y-Z fixed Euler angles of rotation matrix R
        %
        %	XYZ_Euler = Rot2XYZ_Euler(R)
        %
        % R is a rotation matrix. xyz is of the form [theta psi fai]' (around x-,
        % y- and z-axis, respectively).
        % 
        % see http://en.wikipedia.org/wiki/Euler_angles for detail

        psi = atan2(R(1,3), sqrt(R(1,1)^2 + R(1,2)^2));
        if abs(psi - pi/2) < eps
          theta = 0;
          fai = -atan2(R(2,1), R(3,1));
        elseif abs(psi + pi/2) < eps
          theta = 0;
          fai = atan2(R(1,2), R(2,2));
        else
          theta = -atan2(R(2,3)/cos(psi), R(3,3)/cos(psi));
          fai = -atan2(R(1,2)/cos(psi), R(1,1)/cos(psi));
        end

        XYZ_Euler = [theta psi fai]'.*180/pi;
        end

        function matrix = matRotationX(angle_deg)
            % angle_deg can be any size of array
            cos_all = cos(pi/180.0 * angle_deg(:)); sin_all = sin(pi/180.0 * angle_deg(:));
            s = numel(angle_deg);
            matrices = [ones(s,1) zeros(s,4) cos_all sin_all zeros(s,2) -sin_all cos_all zeros(s,4) ones(s,1)];
            matrix = reshape(matrices',[4 4 size(angle_deg)]);
        end

        function matrix = matRotationY(angle_deg)
            % angle_deg can be any size of array
            cos_all = cos(pi/180.0 * angle_deg(:)); sin_all = sin(pi/180.0 * angle_deg(:));
            s = numel(angle_deg);
            matrices = [cos_all zeros(s,1) -sin_all zeros(s,2) ones(s,1) zeros(s,2) sin_all zeros(s,1) cos_all zeros(s,4) ones(s,1)];
            matrix = reshape(matrices',[4 4 size(angle_deg)]);
        end

        function matrix = matRotationZ(angle_deg)
            % angle_deg can be any size of array
            cos_all = cos(pi/180.0 * angle_deg(:)); sin_all = sin(pi/180.0 * angle_deg(:));
            s = numel(angle_deg);
            matrices = [cos_all sin_all zeros(s,2) -sin_all cos_all zeros(s,4) ones(s,1) zeros(s,4) ones(s,1)];
            matrix = reshape(matrices',[4 4 size(angle_deg)]);
        end

        function matrix = matTranslation(x, y, z)
            if(length(x)==3)    % specify by a vector
                matrix = [  1.0 0.0 0.0 x(1);
                            0.0 1.0 0.0 x(2);
                            0.0 0.0 1.0 x(3);
                            0.0 0.0 0.0 1.0 ];
            else
                matrix = [  1.0 0.0 0.0 x;
                            0.0 1.0 0.0 y;
                            0.0 0.0 1.0 z;
                            0.0 0.0 0.0 1.0 ];
            end
        end
        
        function pm = generatePM_Simple(geom)
            numProjections = length(geom.Angles);
            % origin of (u0, v0) is top-left corner of the image
            if(~isfield(geom, 'u0'))    geom.u0 = geom.UV_size(1)/2; end
            if(~isfield(geom, 'v0'))    geom.v0 = geom.UV_size(1)/2; end
            if(length(geom.u0)==1), geom.u0 = repmat( geom.u0, numProjections ); end
            if(length(geom.v0)==1), geom.v0 = repmat( geom.v0, numProjections ); end
            if(~isfield(geom, 'center_offset'))    geom.center_offset = [0 0 0]; end
            % generate projection matrices for simple geometries (one-axis rotation trajectory)
            pm = zeros(3, 4, numProjections);
            for i=1:numProjections
                u0 =   geom.UV_size(1)/2 - geom.u0(i);  % move the origin of u0 to center of the image
                v0 = -(geom.UV_size(2)/2 - geom.v0(i));  % move the origin of v0 to center of the image and flip the direction
                source_detector_base = RegTools.matRotationX(-90) * RegTools.matRotationY(90+geom.Angles(i)) * RegTools.matTranslation(geom.center_offset(1),geom.center_offset(2),geom.center_offset(3));
                DetectorFrame = source_detector_base * RegTools.matTranslation(u0*geom.PixelSize(1), v0*geom.PixelSize(2), -(geom.SDD-geom.SAD));
                S_obj = source_detector_base * [0.0; 0.0; geom.SAD; 1.0];
                S_det = inv(DetectorFrame) * S_obj;
                int = [-S_det(3)/geom.PixelSize(1) 0 S_det(1)+geom.u0(i); 0 -S_det(3)/geom.PixelSize(2) S_det(2)+geom.v0(i); 0 0 1];
                pm(:,:,i) = [int [0;0;0]] * inv([DetectorFrame(1:3,1:3) S_obj(1:3); 0 0 0 1]);
                
                % scale projection matrix so that the pixel size matches
                % (for distance weight in LinearInterpolation back-projection)
                inv_pm3x3 = inv(pm(1:3,1:3,i));
                pm(:,:,i) = pm(:,:,i) * sqrt(sum(inv_pm3x3(1:3,1).^2))/geom.PixelSize(1);
            end
        end
        
        function ProjectionMatrices = generatePM_FromSourcePositions(Src_mm, uv_dim, PixelSize)
            % Generate 3x4 projection matrix (origin ad left-bottom corner, 'pixel' unit)
            % using the specified source positions. Multiple matrices can be generated at once
            % \input
            %   Src_mm (N x 3): source position (x, y, z) with respect to
            %                   detector in mm
            %   uv_dim (1 x 2): u-v dimension of the projection image (pix)
            %   PixelSize (1 x 2): pixel size (mm/pix)
            % \output
            %   ProjectionMatrices (3 x 4 x N): projection matrices
            % 
            % extrinsic = [eye(3) t; 0 0 0 1] = [eye(3) -[Sx; Sy; Sz]; 0 0 0 1];
            % intrinsic = [-Sz/Px 0 Sx/Px; 0 -Sz/Py Sy/Py; 0 0 1];
            % ProjectionMatrices = [intrinsic [0;0;0]] * extrinsic = [intrinsic intrinsic*t]
            numMat = size(Src_mm, 1);
            image_center_pix = repmat(double(uv_dim)/2.0, [numMat 1]) + Src_mm(:,1:2) * diag([1.0 1.0]./PixelSize);
            intrinsics = reshape([-Src_mm(:,3)/PixelSize(1) zeros(numMat,3) -Src_mm(:,3)/PixelSize(2) zeros(numMat,1) image_center_pix ones(numMat,1)]', 3, 3, []);
            t = reshape((-repmat(Src_mm(:,3),1,3).* [-Src_mm(:,1)/PixelSize(1)+image_center_pix(:,1) -Src_mm(:,2)/PixelSize(2)+image_center_pix(:,2) ones(numMat,1)])', 3, 1, []);
            ProjectionMatrices = cat(2, intrinsics, t);
        end
        
        function ProjectionMatrices = generatePM_From9DOFParameters(param9DOF, uv_dim, PixelSize)
            % Generate 3x4 projection matrix (origin ad left-bottom corner, 'pixel' unit)
            % using the specified 9DOF projection parameters 
            % \input
            %   param9DOF (N x 9): (column 1-6: T_detector_object, column 7-9: source position wrt detector)
            %   uv_dim (1 x 2): u-v dimension of the projection image (pix)
            %   PixelSize (1 x 2): pixel size (mm/pix)
            % \output
            %   ProjectionMatrices (3 x 4 x N): projection matrices
            %
            PM = RegTools.generatePM_FromSourcePositions(param9DOF(:,7:9), uv_dim, PixelSize);
            Extrinsics = RegTools.convertTransRotTo4x4_multi(param9DOF(:,1:6)');
            ProjectionMatrices = RegTools.MultiProd(PM, Extrinsics);
        end
        
        function myu_images = HU2Myu(HU_images, myu_water)
            % convert CT images represented in HU to linear attenuation coefficient
            % input:
            %   HU_images: images in HU
            %   myu_water: linear attenuation coefficient of water at the effective
            %              energy in unit of 'mm2/g'. The effective energy
            %              is generally close to 30% or 40% of peak energy
            %               (see http://www.sprawls.org/ppmi2/RADPEN/)
            %              see http://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
            %              for list of attenuation coefficient of water.
            %              (note that the unit of (myu/rho) listed here is 
            %                cm2/g which is cm-1*(cm3/g), and we convert
            %                the unit for myu and rho into mm-1 and mm3/g
            %                respectively, so the value should be
            %                devided by 10)
            %              e.g.
            %               120kVP (peak energy) -> about 40keV (4e-2MeV) (effective energy)
            %                   -> 0.2683cm2/g -> 0.02683mm2/g
            %
            % output:
            %   myu_images: images of myu
            %
            % definition of Hounsfield Unit (air: -1000, water: 0)
            %   HU = 1000 * (myu - myu_water) / myu_water
            %  ->
            %   myu = HU * myu_water / 1000 + myu_water

            % we clamp negative value
            myu_images = max( (1000+single(HU_images)) * myu_water / 1000, 0 );
        end
      
        function [intrinsics F_Object_Cameras scales] = DecomposePrmMatrices( prm_matrices )
            % Decompose 3x4 camera projection matrix using QR decomposition and 
            % compute intrinsic & extrinsic parameters
            % \input file name
            % \output   intrinsics: intrinsic parameters (3x3xn matrix)
            %           F_Object_Cameras: homogeneous transformation matrix from camera
            %                             coordinate to the object (CT) coordinate
            %                             (4x4xn matrix)
            %           scales: scale factor for intrinsic parameters (1xn matrix)
            %           (n: number of projections)
            %

            intrinsics = [];
            F_Object_Cameras = [];
            scales = [];
            for j=1:size(prm_matrices, 3);
                % Decompose camera projection matrix
                % Note that we want (upper triangular matrix) * (orthonormal matrix)
                % which is opposite order to QR decomposition.
                % We use the following relation: inv(QR) = inv(R) * inv(Q).
                % See, for example,
                % http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node5.html
                % for detail of the decomposition of camera projection matrix
                [Q, R] = qr(inv(prm_matrices(1:3,1:3,j)));
                intrinsic = inv(R);
                extrinsic = Q';

                % We change the sign and scale of the upper triangular matrix 
                % in order to make it appropriate form of intrinsic parameters
                % (diagonal elements are both negative and (3,3) element is positive one)
                for i=1:3
                    if((i<3 && intrinsic(i,i)>0) || (i==3 && intrinsic(3,3)<0))
                        intrinsic(:,i) = -intrinsic(:,i);
                        extrinsic(i,:) = -extrinsic(i,:);
                    end
                end
                scale = intrinsic(3,3);
                intrinsic = intrinsic ./ scale;

                % if the extrinsic parameter is left handed, we change the sign to make
                % it right handed
                if(extrinsic(:,3)' * cross(extrinsic(:,1),extrinsic(:,2)) < 0)
                    extrinsic = extrinsic * -1;
                    scale = scale * -1;
                end

                % translation part of the extrinsic parameter can be computed by
                % multiplying inverse of the intrinsic parameter and 4th column of
                % the camera projection matrix
                % P = s C [ R | T ] -> T = inv(C) * (P(:,4) / s)
                translation = inv(intrinsic) * prm_matrices(:,4,j) ./ scale;

                % store return variables
                intrinsics = cat(3, intrinsics, intrinsic);
                F_Object_Cameras = cat(3, F_Object_Cameras, inv([extrinsic, translation; [0, 0, 0, 1]]));
                scales = cat(2, scales, scale);
            end
        end
        
        function [Rigid Affine] = DecomposeAffineTransform( in_affine )
            % decompose an affine transform into rigid transform + affine transform
            [Rigid Affine] = qr(in_affine);
            Rigid(1:3,4) = Rigid(1:3,1:3)*Affine(1:3,4);
            Affine(1:3,4) = 0;
            for i=1:3
                if(Rigid(i,i)<0)
                    Rigid(:,i) = -Rigid(:,i);
                    Affine(i,:) = -Affine(i,:);
                end
            end
        end
        
        function [Affine Rigid] = DecomposeAffineTransform_inv( in_affine )
            % decompose an affine transform into affine transform + rigid transform
            [invRigid invAffine] = qr(inv(in_affine));
            invRigid(1:3,4) = invRigid(1:3,1:3)*invAffine(1:3,4);
            invAffine(1:3,4) = 0;
            Affine = inv(invAffine);
            Rigid = inv(invRigid);
            for i=1:3
                if(Rigid(i,i)<0)
                    Affine(:,i) = -Affine(:,i);
                    Rigid(i,:) = -Rigid(i,:);
                end
            end
        end
        
        function PM_downsampled = DownSampleProjectionMatrices( PM, down_sample_ratio )
            % PM: 3x4xN
            % down_sample_ratio: scalar
            
            [intrinsics, F_Object_Cameras, scales] = RegTools.DecomposePrmMatrices( PM );
            intrinsics(1,1,:) = intrinsics(1,1,:)/down_sample_ratio;
            intrinsics(2,2,:) = intrinsics(2,2,:)/down_sample_ratio;
            intrinsics(1:2,3,:) = intrinsics(1:2,3,:)/down_sample_ratio;
            PM_downsampled = zeros(size(PM));
            for i=1:size(PM,3)
                PM_downsampled(:,:,i) = scales(i) * [intrinsics(:,:,i) [0;0;0]] / F_Object_Cameras(:,:,i);
            end
        end
        
        % utility functions to make block multiplication (a set of 4x4 matrices) more efficient
        % this eliminates for loop, thus improves performance
        % e.g.,
        %     A: 4x4 matrix, B: 4x4xN matrix array, C: 4x4 matrix
        %     > mult = reshape( A*reshape(B,4,[]), 4,4,[] );
        %     --> mult(:,:,1) = A*B(:,:,1), mult(:,:,2) = A*B(:,:,2), ...
        % 
        %     > mult = reshape( BlockTranspose_4Nx4(BlockTranspose_4x4N(reshape(B,4,[])) * C), 4,4,[] );
        %     --> mult(:,:,1) = B(:,:,1)*C, mult(:,:,2) = B(:,:,2)*C, ...
        %
        %     > mult = reshape( BlockTranspose_4Nx4(BlockTranspose_4x4N(A*reshape(B,4,[])) * C), 4,4,[] );
        %     --> mult(:,:,1) = A*B(:,:,1)*C, mult(:,:,2) = A*B(:,:,2)*C, ...
        function transposed = BlockTranspose_4x4N( mat_4x4N )
            transposed = reshape( permute( reshape( mat_4x4N,4,4,[] ), [1 3 2] ),[],4 );
        end
        function transposed = BlockTranspose_4Nx4( mat_4Nx4 )
            transposed = reshape( permute( reshape( mat_4Nx4',4,4,[] ), [2 1 3] ),4,[] );
        end
        
        function Output = MultiProd(M, X, no_X_permute)
            % Multiple matrix product for systems having the same size
            % (vectorized version)
            %
            % \input
            %   M  : 3D array (m x n x p)
            %   X  : 3D array (n x q x p)
            % \output
            %   Output: 3D array (m x q x p)
            %
            % Perform p matrix products:
            %   Output(:,:,k) = M(:,:,k) * X(:,:,k), for all k=1,2,...,p
            %

            % error check
            if(ndims(M)>3 || ndims(X)>3), error('MultiProd: M or X cannot have more than 3 dimensions'); end
            [m, n, p] = size(M);
            if(size(X,1)~=n), error('MultiProd: M and X dimensions are not compatible'); end

            % Build sparse matrix and solve
            if(~exist('no_X_permute','var') || no_X_permute == 0)
                X = permute(X,[1 3 2]);
            end
            X = reshape(X, n*p, []); % (n*p) x q
            I = repmat(reshape(1:m*p,m,1,p),[1 n 1]); % m x n x p
            J = repmat(reshape(1:n*p,1,n,p),[m 1 1]); % m x n x p
            A = sparse(I(:),J(:),M(:));
            Output = permute( reshape(A * X, [m p size(X,2)]), [1 3 2]);
        end
        
        function p_6xN = PrePostMultiplyTranslation( p_6xN, t )
            % multiply +t from left side and -t from right side, i.e.,
            % output = [eye(3) t; 0 0 0 1] * [R T; 0 0 0 1] * [eye(3) -t; 0 0 0 1]
            %        = [R T+t-R*t; 0 0 0 1]
            % each column of p_6xN is represented as [tx, ty, tz, rx, ry, rz] 
            % where (tx,ty,tz) is translation and (rx,ry,rz) is rotation (in 'degree').
            % Computation of multiple R*t is vectorized. Rotation is equivalent to
            % RegTools.matRotationZ(p_6xN(6)) * RegTools.matRotationY(p_6xN(5)) * RegTools.matRotationX(p_6xN(4))
            angle_rad = pi/180.0 * p_6xN(4:6,:);
            c = cos(angle_rad); s = sin(angle_rad);
            Rt = [ (c(2,:).*c(3,:))*t(1) + (-c(1,:).*s(3,:)+s(1,:).*s(2,:).*c(3,:))*t(2) + ( s(1,:).*s(3,:)+c(1,:).*s(2,:).*c(3,:))*t(3);
                   (c(2,:).*s(3,:))*t(1) + ( c(1,:).*c(3,:)+s(1,:).*s(2,:).*s(3,:))*t(2) + (-s(1,:).*c(3,:)+c(1,:).*s(2,:).*s(3,:))*t(3);
                       -s(2,:)*t(1)      + (s(1,:).*c(2,:))*t(2)                         + ( c(1,:).*c(2,:))*t(3)];
            p_6xN(1:3,:) = p_6xN(1:3,:) + t(:)*ones(1,size(p_6xN,2)) - Rt;            
        end
        

        function p_6xN = ChangeRotationCenter( p_6xN, t )
            
            p_4x4xN = RegTools.convertTransRotTo4x4_multi(p_6xN) ;
            
            Transe = [eye(3) -t(:); 0 0 0 1] ;
            Transe = repmat( Transe, 1, 1, size(p_4x4xN,3)) ;
            
            p_4x4xN= RegTools.MultiProd(p_4x4xN, Transe) ;
            
            Transe = [eye(3) t(:); 0 0 0 1] ;
            Transe = repmat( Transe, 1, 1, size(p_4x4xN,3)) ;
            
            p_4x4xN= RegTools.MultiProd(Transe, p_4x4xN) ;
            
            p_6xN = RegTools.convert4x4ToTransRot_multi(p_4x4xN) ;

        end

        function p_4x4xN = PrePostMultiplyTranslation_4x4( p_4x4xN, t )
            Trans = repmat( [eye(3) -t(:); 0 0 0 1], 1, 1, size(p_4x4xN,3)) ;
            p_4x4xN= RegTools.MultiProd(p_4x4xN, Trans) ;
            Trans = repmat( [eye(3) t(:); 0 0 0 1], 1, 1, size(p_4x4xN,3)) ;
            p_4x4xN= RegTools.MultiProd(Trans, p_4x4xN) ;
        end
      
        function [affineWarpX, affineWarpY, affineWarpZ] = convertAffineToWarp( affineTransformation, nx, ny, nz )
            % compute deformation field that is equivallent to an affine transform
            [X, Y, Z] = meshgrid(1:nx, 1:ny, 1:nz);
            X = permute(X, [2 1 3])-nx/2; Y = permute(Y, [2 1 3])-ny/2; Z = permute(Z, [2 1 3])-nz/2;
            warped = affineTransformation(1:3,:) * [X(:)'; Y(:)'; Z(:)'; ones(1,numel(X))];
            affineWarpX = reshape(warped(1,:)-X(:)', [nx ny nz]);
            affineWarpY = reshape(warped(2,:)-Y(:)', [nx ny nz]);
            affineWarpZ = reshape(warped(3,:)-Z(:)', [nx ny nz]);
        end
    end
end
