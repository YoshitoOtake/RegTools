classdef Registration2D3DInterface < hgsetget & handle
    %REGISTRATION2D3DINTERFACE - Interface for any 2D/3D registration
    %algorithm.
    %
    %   This class represents the interface for any 2D3D registration
    %   routine. This is the external API that is presented to a C/C++
    %   caller. All 2D3D registration techniques should have a wrapper that
    %   implements this API to ease the integration between C/C++ and
    %   Matlab.
    %
    %   The abstract methods are those that need to be implemented in the
    %   child classes. The set.Property methods have been implemeneted for
    %   all properties already, these should likely not be overridden
    %   unless you have a good reason. These methods do lots of argument
    %   checking and make sure everything is valid. For those unfamiliar
    %   with matlab, these matlab-based set methods are called
    %   automatically anytime a property value is set. As such, when
    %   calling SetInitialGuess, the child class MUST having something
    %   like:
    %       obj.InitialGuess = initalGuess;
    %   to ensure that the inital guess is stored in the object. However,
    %   there is no need to ensure the validity of the argument; one can
    %   assume it is valid. If the argument is invalid, no change will be
    %   made to the property.
    %
    %   To subclass in Matlab:
    %       classdef ChildClass < ParentClass
    %
    %   In this case, please use:
    %       classdef ChildClass < Registration2D3DInterface & handle
    %   The '& handle' serves as a reference to Matlab that you're
    %   developing a handle class. This doesn't have any influence on the
    %   runtime, but will help with MLint and pointing out errors/conflicts
    %   with handle class usage.
    %
    %   When developing the subclass, you'll need to follow the block
    %   structure that is outlined below (i.e., use something is in a
    %   protected block, use that). The exception is when you implmenet
    %   abstract methods. Use all the same parameters except drop the
    %   Abstract designation and that should work.
    %
    %   This class assumes we will be using a projection matrix to define
    %   the camera geometry. This is a 3x4 matrix combining both intrinsic
    %   and extrinsic parameters. For more details, please see the
    %   PROJECTIONGEOMETRY class.
    %
    %   Registration2D3DInterface properties:
    %       INITIALGUESS - the initial guess
    %       PROJECTIONMATRICES - the input projection matrices
    %       IMAGES - the input images
    %       REGISTRATIONTRANSFORM - the registration transform
    %       MODEL - the input model
    %       REGISTRATIONTIME - Elapsed time for registration trial
    %       PARAM - Registration parameters
    %
    %       INPUTVOLUME - the input volume
    %
    %   Registration2D3DInterface methods:
    %       GETREGISTRATION - get the registration
    %       SETINPUT - Set the input parameters
    %
    %   Registration2D3DInterface abstract methods:
    %       SETINITIALGUESS - set the initial guess
    %       SETPROJECTIONMATRICES - set the projection matrices
    %       SETIMAGES - set the 2d images
    %       RUN - Run the process
    %
    %   Registration2D3DInterface static methods:
    %       CHECKMODEL - Check a model and compute normals if necessary
    %       CHECK4X4 - Check a 4x4 homogeneous matrix
    %
    %   See also: PROJECTIONGEOMETRY.
    
    %%%%%%%%%%%%%%%%%%%%%%
    % Default properties %
    %%%%%%%%%%%%%%%%%%%%%%
    properties
        %INITIALGUESS - The initial registration guess. This should be a
        %4x4 homogeneous matrix.
        InitialGuess;
        
        %PROJECTIONMATRICES - The camera projection matrices. This is a
        %3x4xnImages matrix.
        ProjectionMatrices = zeros(3, 4);
        
        %IMAGES - The images to perform the registration with. This is an
        %mxnxnImages.
        Images;
        
        %PIXELSIZE - Pixel size of the input images. This is a 1x2 array of
        %double.
        PixelSize;
        
        %REGISTRATIONTRANSFORM - The final registration transformation.
        %This should be populated during the RUN method
        RegistrationTransform;
        RegistrationTransform_4x4;
       
        %PARAM - The registration parameters
        Param;
    end % properties
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Protected set properties %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties (SetAccess = protected)
        % subclasses CAN modify these variables.
        
        %INPUTVOLUME - The input image volume. This is a m x n x p matrix
        %of singles.
        InputVolume;
        
        %VOXELSIZE - Voxel size of the input volume. This is a 1x3 array
        %double.
        VoxelSize;
        
        %REGISTRATIONTIME - Elapsed time for one registration trial
        RegistrationTime; % sec
    end % Protected set access
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Private set properties %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    properties (SetAccess = private)
        % Subclasses cannot modify these values, only get the values. To
        % modify, you'll need to set the model.
        
        %FACES - The faces of a surface model. This is a 3 x nFaces
        Faces;
        
        %VERTICES - The vertices of a surface model. The is a 3 x nVertices
        Vertices;
        
        %FACENORMALS - The face normals of a surface model. 3 x nFaces     
        FaceNormals;
        
        %VERTEXNORMALS - The vertex normals of a surface model. 3 x
        %nVertices
        VertexNormals;
    end % Private set access
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % Dependent properties %
    %%%%%%%%%%%%%%%%%%%%%%%%
    properties ( Dependent = true )
        % This is a dependent property. You can explicity set/get the
        % values, but the actual represntation is never stored.
        
        %MODEL - The surface model. This is a structure composed of faces 
        % and vertices. Setting the model automatically sets the
        % Faces/Vertices and associated normals
        Model;
    end % dependent properties
    
    %%%%%%%%%%%%%%%%%%%%
    % Abstract methods %
    %%%%%%%%%%%%%%%%%%%%
    methods (Abstract)
        % Abstract methods. These methods MUST be implemented in
        % subclasses, otherwise Matlab will error.
        
        %SETINITIALGUESS - Set the inital guess. The guess should be a
        %column vector.
        %
        %   obj.SetInitialGuess(guess);
        SetInitialGuess(obj, guess);

        %SETPROJECTIONMATRICES - Set the projection matrix. This should
        %be a 3x4xnImages matrix.
        %
        %   obj.SetProjectionMatrices(pms);
        SetProjectionMatrices(obj, pms);
        
        %SETIMAGES - Set the images. The input image should be an nCols x
        %nRows x nImages matrix (e.g. the transpose of the matlab
        %representation). As such, this algorithm should appropriately
        %handle this case.
        %
        %   obj.SetImages(imagse);
        SetImages(obj, images);
        
        %SETPIXELSIZE - Set the pixel size. The input  should be an 1 x
        %2 array
        %
        %   obj.SetPixelSize(pixel_size);
        SetPixelSize(obj, pixel_size);
                
        %SETVOXELSIZE - Set the voxel size
        %
        %   obj.SetVoxelSize(voxel_size)
        SetVoxelSize(obj, voxel_size);

        %RUN - Run the registration algortihm
        %
        %   obj.Run();
        Run(obj);
        
        %INITIALIZE - Initialize the registration procedure
        %
        %   obj.Initialize();
        Initialize(obj);
    end % Abstract methods
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Protected abstract methods %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Abstract, Access = protected)
        % Protected abstract methods. These methods must be implemented in
        % subclasses. Note that these are not available to an external
        % caller.
        
        %SETVOLUME - Set the volume
        %
        %   obj.SetVolume(volume)
        SetVolume(obj, volume);
        
        %SETMODEL - Set the model
        %
        %   obj.SetModel(model);
        SetModel(obj, model);
    end % abstract, protected methods
    
    %%%%%%%%%%%%%%%%%%
    % Public methods %
    %%%%%%%%%%%%%%%%%%
    methods
        % Public methods. These are available to external callers.
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Matlab-based Set Methods %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Override if needed, but these methods check whenever a class
        % variable is set. So, for instance, if you type 
        %       InitialGuess = I
        % in a function, the set.InitialGuess method is automatically
        % called.
        function set.InitialGuess(obj, guess)
            %SET.INITIALGUESS - Set the initial guess. 
            %
            % This is the Matlab-based implementation that ensures the
            % inital guess is of an appropriate form
            % (new feature: initial guess should be a column vector)
            
            if ~iscolumn(guess)
                fprintf(2, 'Could not set initial guess\n');
                return;
            end
            
            obj.InitialGuess = guess;
        end % set.InitialGuess
        function set.InputVolume(obj, volume)
            %SET.INPUTVOLUME - Set the input volume
            %
            %
            %   This checks to make sure it's real numbers and a 3dim
            %   matrix
            
            if iscell(volume)
                % volume can be a cell array of multiple volumes
                obj.InputVolume = volume;
                return;
            end
            
            if ~isnumeric(volume) || ...
                ~isreal(volume) || ...
                any(isinf(volume(:))) || ...
                any(isnan(volume(:))) || ...
                ndims(volume) ~= 3
                fprintf(2, 'Could not set input volume\n');            
                return;
            end
            
            obj.InputVolume = volume;
            
        end % set.InputVolume
        function set.PixelSize(obj, pixel_size)
            %SET.PIXELSIZE - Set the pixel size
            %
            %
            %   This checks to make sure it's real numbers and a 2dim
            %   matrix
            
            if ~isnumeric(pixel_size) || ...
                ~isreal(pixel_size) || ...
                any(isinf(pixel_size(:))) || ...
                any(isnan(pixel_size(:))) || ...
                length(pixel_size) ~= 2
                fprintf(2, 'Could not set pixel size\n');            
                return;
            end
            
            obj.PixelSize = pixel_size;
            
        end % set.InputVolume
        function set.VoxelSize(obj, voxel_size)
            %SET.VOXELSIZE - Set the voxel size
            %
            %
            %   This checks to make sure it's real numbers and a 3dim
            %   matrix
            
            if ~isnumeric(voxel_size) || ...
                ~isreal(voxel_size) || ...
                any(isinf(voxel_size(:))) || ...
                any(isnan(voxel_size(:)))
%             || length(voxel_size) ~= 3
                fprintf(2, 'Could not set voxel size\n');            
                return;
            end
            
            obj.VoxelSize = voxel_size;
            
        end % set.VoxelSize
        function set.ProjectionMatrices(obj, pms)
            %SET.PROJECTIONMATRICES - Set the projection matrices
            
            if ~isreal(pms) || ...
                    ~isnumeric(pms) || ...
                    any(isinf(pms(:))) || ...
                    any(isnan(pms(:))) || ...
                    ndims(pms) <= 1 || ...
                    ~isequal(size(pms(:,:,1)), [3 4])
                fprintf(2, 'Could not set projection matrix\n');
                return;
            end
            
            obj.ProjectionMatrices = pms;
            
        end % set.ProjectionMatrices
        function set.Images(obj, images)
            %SET.IMAGES - Set the image(s).
            
            if ~isreal(images) || ...
                    ~isnumeric(images) || ...
                    ndims(images) <= 1 || ...
                    any(isinf(images(:))) || ...
                    any(isnan(images(:))) 
                fprintf(2, 'Could not set images\n');
                return;
            end
            
            obj.Images = images;
        end % set.Images
        function set.Model(obj, model)
            %SET.MODEL - Set the model.
            
            [isValid m] = obj.CheckModel(model);
            
            if ~isValid
                fprintf(2, 'Could not set model\n');
                return;
            end
                
            obj.Faces = m.Faces;
            obj.Vertices = m.Vertices;
            obj.FaceNormals = m.FaceNormals;
            obj.VertexNormals = m.VertexNormals;
        end % set.Model
        
        function val = hasInputVolume(obj)
            val = ~isempty(obj.InputVolume);
        end
        
        function val = hasImages(obj)
            val = ~isempty(obj.Images);
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Matlab-based Get Methods %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function model = get.Model(obj)
            %GET.MODEL - Get a structure representation of the model, if it
            %exists. Otherwise, return empty
            
            if isempty(obj.Faces) || isempty(obj.Vertices)
                model = [];
            else
                model.Faces = obj.Faces;
                model.Vertices = obj.Vertices;
                model.FaceNormals = obj.FaceNormals;
                model.VertexNormals = obj.VertexNormals;
            end
        end % get.Model
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % C++ Interface functions %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Tform = GetRegistration(obj)
            %GETOUTPUT - Get the registration result
            %
            %   Override this method in subclasses to change the number of
            %   output arguments, if necessary.
            %
            %   Tform = obj.GetRegistration();
            
            Tform = obj.RegistrationTransform;
            
        end % GetOutput

        function vec1x6 = GetRegistration_vec1x6(obj)
            vec1x6 = RegTools.convert4x4ToTransRot(obj.RegistrationTransform);
        end % GetRegistration_vec1x6
        
        function SetInput(obj, varargin)
            %SETINPUT - Set the input.
            %
            %   obj.SetInput(volume) - set an input volume
            %   The input volume should be an m x n x p matrix of singles.
            %   (new feature: volume can be a cell array of multiple single
            %   matrices)
            %
            %   obj.SetInput(faces, vertices) - set an input model and
            %       compute face/vertex normals
            %   obj.SetInput(model) - set an input model, compute
            %       face/vertex normals if necessary
            %   obj.SetInput(faces, vertices, faceNormals, vertexNormals) -
            %       set an input model using the given normals
           
            if nargin == 1
                obj.SetVolume();
            elseif nargin == 2
                arg1 = varargin{1};
                if isstruct(arg1)
                    obj.SetModel(arg1);
                elseif isnumeric(arg1) || iscell(arg1)
                    obj.SetVolume(arg1);
                end
            elseif nargin == 3
                m.Faces = varargin{1};
                m.Vertices = varargin{2};
                
                obj.SetModel(m);
            elseif nargin == 5
                m.Faces = varargin{1};
                m.Vertices = varargin{2};
                m.FaceNormals = varargin{3};
                m.VertexNormals = varargin{4};
                
                obj.SetModel(m);
            end
        end % SetInput
    end % methods
    
    %%%%%%%%%%%%%%%%%%
    % Static methods %
    %%%%%%%%%%%%%%%%%%
    methods (Static)
        function is4x4 = Check4x4(tform)
            %CHECK4X4 - Check the validity of a 4x4 transform
            
            is4x4 = false;
            tol = 1e-3;

            try 
                
                dI = tform(4,:) - [0 0 0 1];
                dR = tform(1:3,1:3)*tform(1:3,1:3)' - eye(3);
            
                % check:
                %   numeric
                %   4x4
                %   real
                %   no infs
                %   no nans
                %   det == 1, within 1e-6
                %   last row is [0 0 0 1]
                %   valid rotation
                if ~isnumeric(tform)

                elseif ~isequal(size(tform), [4 4])
                    
                elseif ~isreal(tform)
                    
                elseif any(isinf(tform(:)))
                    
                elseif any(isnan(tform(:)))
                        
                elseif abs(det(tform) - 1) > tol
                    
                elseif any(abs(dI(:)) > tol)
                    
                elseif any(abs(dR(:)) > tol)
                    
                else
                    is4x4 = true;
                end
                
            catch ME
                return
            end
        end % Check4x4 matrix
        
        function [isValid m] = CheckModel(model)
            %CHECKMODEL - Check the validity of a model. A model should be
            %a structure composed of faces and vertices (case insensitive).
            %The vertices should be 3xnVertices and the faces should be
            %3xnFaces. Faces must be indexed [1, nVertices]. The model
            %should also include FaceNormals (3xnFaces) and VertexNormals
            %(3xnVertices). In the event that these are missing, the face
            %and vertex normals will be computed.
            %
            %   Note, the following are equivalent naming conventions
            %       'Faces' and 'ElementData'
            %       'Vertices' and 'NodeData'
            %       'FaceNormals' and 'ElementNormals'
            %       'VertexNormals' and 'NodeNormals'
            
            m.Faces = [];
            m.Vertices = [];
            m.FaceNormals = [];
            m.VertexNormals = [];
            
            isValid = false;                       
            
            try
            
                if ~isstruct(model)
                    return;
                end
                
                names = fieldnames(model);
                if numel(names) < 2															% rev. Ben
                    return;
                end
                
                % check struct files and pull out appropriate data
                foundFaces = false;
                foundVerts = false;
                foundFaceNormals = false;
                foundVertexNormals = false;
                for i=1:numel(names)														% rev. Ben
                    switch(lower(names{i}))
                        case {'faces' 'elementdata'}
                            m.Faces = model.(names{i});
                            foundFaces = true;
                        case {'vertices' 'nodedata'}
                            m.Vertices = model.(names{i});                        
                            foundVerts = true;
                        case {'facenormals' 'elementnormals'}
                            m.FaceNormals = model.(names{i});
                            foundFaceNormals = true;
                        case {'vertexnormals' 'nodenormals'}
                            m.VertexNormals = model.(names{i});
                            foundVertexNormals = true;
                    end
                end
                if ~(foundFaces && foundVerts)
                    m = [];
                    return;
                end
                
                % check vertices
                [a nVerts] = size(m.Vertices);
                
                if a ~= 3 && nVerts == 3
                    nVerts = a;
                    a = 3;
                    m.Vertices = m.Vertices';
                end

                if ~isnumeric(m.Vertices) || ...
                        ~isreal(m.Vertices) || ...
                        any(isinf(m.Vertices(:))) || ...
                        any(isnan(m.Vertices(:))) || ...
                        ndims(m.Vertices) ~= 2 || ...
                        a ~= 3
                    return;
                end
                
                % check faces
                [a nFaces] = size(m.Faces);
                if a~=3 && nFaces == 3
                    nFaces = a;
                    a = 3;
                    m.Faces = m.Faces';
                end
                if ~isnumeric(m.Faces) || ...
                        ~isreal(m.Faces) || ...
                        any(isinf(m.Faces(:))) || ...
                        any(isnan(m.Faces(:))) || ...
                        ndims(m.Faces) ~= 2 || ...
                        a ~= 3
                    return;
                end
                
                % check face indices
                minF = min(m.Faces(:));
                maxF = max(m.Faces(:));
                
                if minF ~= 1 || maxF ~= nVerts
                    return;
                end
                
                % ensure it's only integers
                if ~isequal(m.Faces, int32(m.Faces))
                    return;
                end
                
                if ~foundFaceNormals
                    % compute face normals here
                    
                    v1 = m.Vertices(:, m.Faces(1,:));
                    v2 = m.Vertices(:, m.Faces(2,:));
                    v3 = m.Vertices(:, m.Faces(3,:));
                    
                    v12 = v2-v1;
                    v13 = v3-v1;

                    m.FaceNormals = Registration2D3DInterface.Normalize(...
                        cross(v12, v13) );

                else
                    % check validity of face normals
                    [a nFNorms] = size(m.FaceNormals);
                    if a ~=3 && nFNorms == 3
                        nFNorms = a;
                        a = 3;
                        m.FaceNormals = m.FaceNormals';
                    end
                    if ~isnumeric(m.FaceNormals) || ...
                            ~isreal(m.FaceNormals) || ...
                            any(isinf(m.FaceNormals(:))) || ...
                            any(isnan(m.FaceNormals(:))) || ...
                            ndims(m.FaceNormals) ~= 2 || ...
                            nFNorms ~= nFaces || ...
                            a ~= 3
                        return;
                    end
                    
                end
                
                if ~foundVertexNormals
                     % compute vertex normals here

                    meshConnectivity = ...
                        Registration2D3DInterface.MeshConnectivity(m.Faces', nVerts);
                    m.VertexNormals = ...
                        Registration2D3DInterface.AverageNormals(...
                        meshConnectivity,...
                        m.FaceNormals, ...
                        nVerts);
                else
                    % check validity of vertex normals
                    [a nVNorms] = size(m.VertexNormals);
                    if a~=3 && nVNorms == 3
                        nVNorms = a;
                        a = 3;
                        m.VertexNormals = m.VertexNormals';
                    end
                    if ~isnumeric(m.VertexNormals) || ...
                            ~isreal(m.VertexNormals) || ...
                            any(isinf(m.VertexNormals(:))) || ...
                            any(isnan(m.VertexNormals(:))) || ...
                            ndims(m.VertexNormals) ~= 2 || ...
                            nVNorms ~= nVerts || ...
                            a ~= 3
                        return;
                    end
                end
            catch ME
                keyboard
                return;
            end
            
            isValid = true;
        end % CheckModel
        
        function [vertexToFaceConnectivity vertexConnectivity] = ...
                MeshConnectivity( elementData, nVertices )
            % MESHCONNECTIVITY - Compute the mesh connecitivity for a model
            % This is taken from f_mesh_connectivity in the BGS suite.
                        
            nFaces = size(elementData,1);
            
            % sparse logical indices identifying faces that are connected to each
            % vertex
            idx = (1:nFaces)';
            vertexToFaceLookup = [[idx; idx; idx;] elementData(:)];
            vertexToFaceLookup = unique(vertexToFaceLookup,'rows');
            vertexToFaceConnectivity = sparse(...
                vertexToFaceLookup(:,1),...
                vertexToFaceLookup(:,2),...
                true,nFaces,nVertices);

            if nargout > 1
                % this connectivity is typically not needed
                allEdges = sortrows(sort(...
                    [elementData(:,[1 2]); ...
                    elementData(:,[1 3]); ...
                    elementData(:,[2 3])],2));
                connectedPoints = unique(allEdges,'rows');
                % if point i is connected with point j then j is connected with i as well
                connectedPoints = ...
                    [connectedPoints; fliplr(connectedPoints)];

                vertexConnectivity = sparse(...
                    connectedPoints(:,1),...
                    connectedPoints(:,2),...
                    true(length(connectedPoints),1),nVertices,nVertices);
            end
        end % MeshConnectivity
        function [nodeUnitNormals] = ...
                AverageNormals(meshConnectivity, faceNormals, nNodes)
            %AVERAGENORMALS - Average the normals
            %
            % This is taken from f_average_normals
            
            nodeVectors = zeros(3,nNodes);
            for iNode = 1:nNodes
                idx = meshConnectivity(:,iNode);
                % Note this should be the vector sum of unit normals that gets
                % normalized later.  The code commented out is some kind of average
                % that doen't make much sense
                nodeVectors(:,iNode) = sum(faceNormals(:,idx),2);
            end

            nodeUnitNormals = Registration2D3DInterface.Normalize(nodeVectors);
        end % AverageNormals
        function norms = Normalize( a )
            %NORMALIZE - Normalize a 3xn matrix of column vectors
        
            b = sum(a.^2).^0.5;
            c = repmat(b,3,1);
            c(c == 0) = eps;
            norms = a ./ c;
        end % Normalize
        function ProjectionDistance_pix = ComputeProjectionDistance(Target3Ds_mm, ProjectionMatrix, TrueTarget2Ds_pix)
            % compute mean projection distance error (in pixels) between projection of
            % Target3Ds (n x 3 matrix) and the true 2D positions of the
            % target TrueTarget2Ds (n x 2 matrix)
            numTarget = size(Target3Ds_mm, 1);
            pr = ProjectionMatrix * [Target3Ds_mm'; ones(1, numTarget)];
            projected = [pr(1,:)./pr(3,:); pr(2,:)./pr(3,:)]';
            ProjectionDistance_pix = mean(sqrt(sum((projected - TrueTarget2Ds_pix).^2,2)),1);
        end % ComputeProjectionDistance      
    end % Static methods
    
end % Registration2D3DInterface