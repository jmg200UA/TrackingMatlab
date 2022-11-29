% Creamos el detector de caras
faceDetector = vision.CascadeObjectDetector();

% Creamos el tracker de puntos
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Leemos y alamcenamos el video
cam =  VideoReader('caras_1.avi');

% Capturamos un fotograma para obtener su tamaño
videoFrame = readFrame(cam);
frameSize = size(videoFrame);

% Creamos el objeto videoPlayer
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);


% Inicializamos las variables
runLoop = true;
numPts = 0;
video = VideoWriter('faces.mp4'); %create the video object
open(video);
while runLoop

    % Lectura del frame y binarización para detección de puntos
    videoFrame = readFrame(cam);
    videoFrameGray = rgb2gray(videoFrame);

    if numPts < 10
        % Modo de detección
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            % Encontramos los puntos dentro de la región detectada
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % Volvemos a inicializar el tracker de puntos
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Guardamos una copia de los puntos anteriores
            oldPoints = xyPoints;
            
            % Convertimos el rectangulo representado por [ejeX, ejeY, ancho,
            % alto] en una matriz de N x 2.
            % Esto lo hacemos para poder transformar el cuadro de selección para la
            % orientación de la cara
            bboxPoints = bbox2points(bbox(1, :));
            
            %Convertimos las esquinas del cuadro en [x1 y1 x2 y2 x3 y3 x4 y4]
            % Esto es necesario para poder hacer el insertShape
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Insertamos el cuadro alrededor de la cara detectada
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Mostramos los puntos detectados
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
        % Tracking
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % Estimamos la tranformación geométrica entre los puntos
            % antiguos y los nuevos
            [xform, inlierIdx] = estimateGeometricTransform2D(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            oldInliers    = oldInliers(inlierIdx, :);
            visiblePoints = visiblePoints(inlierIdx, :);

            % Aplicamos la transformación al cuadro de selección
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Como hicimos anteriormente convertimos las esquinas del cuadro en [x1 y1 x2 y2 x3 y3 x4 y4]
            % Esto es necesario para poder hacer el insertShape
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Insertamos el cuadro alrededor de la cara detectada
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Mostramos los puntos detectados
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
             writeVideo(video,videoFrame);

            % Reseteamos los puntos
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

    end

    % Mostramos el fotograma de video correspondiente
   
    step(videoPlayer, videoFrame);
    
    % Comprobamos si de ha cerrado la ventana del reproductor de video
    runLoop = isOpen(videoPlayer);
end

% Limpiamos los objetos
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);
close(video);