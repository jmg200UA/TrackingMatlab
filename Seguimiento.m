videoReader = VideoReader('caras_1.avi');
videoPlayer = vision.VideoPlayer('Position',[100,100,680,520]);
objectFrame = readFrame(videoReader);

figure; imshow(objectFrame);

objectRegion=round(getPosition(imrect));

    bboxPoints = bbox2points(objectRegion(1, :));
    bboxPolygon = reshape(bboxPoints',1,[]);
objectImage = insertShape(objectFrame,'Polygon',bboxPolygon,'Color','red');
figure;
imshow(objectImage);
title('Caja del Objeto');
%roiRect = imrect;

points = detectMinEigenFeatures(im2gray(objectFrame),'ROI',objectRegion);
pointImage = insertMarker(objectFrame,points.Location,'+','Color','white');
figure;
imshow(pointImage);
title('Puntos de interes');

tracker = vision.PointTracker('MaxBidirectionalError', 50);
initialize(tracker,points.Location,objectFrame);


video = VideoWriter('yourvideo.mp4'); %create the video object
open(video); %open the file for writing

while hasFrame(videoReader)
        frame = readFrame(videoReader);
      [points,validity] = tracker(frame);
      
      pointSize= size(points);
      sizePoints= pointSize(1);
      %prueba= sizePoints/2;
      puntox=100000000;
      puntoy=100000000;
      ymax=-1;
      xmax=-1;
      recorte= size(frame);
      
      for i=1 :sizePoints
          if (puntox >points(i,1) && points(i,1)>recorte(2)/13)
              puntox=points(i,1);
          end
        if xmax < points(i,1)
            xmax=points(i,1);
        end
        if (puntoy > points(i,2))
            puntoy=points(i,2);
        end
        if ymax < points(i,2)
            ymax=points(i,2);
        end
            
      end
      

      
      out = insertMarker(frame,points(validity, :),'+');
      frame1 = imcrop(out, [puntox puntoy 700 700]);
      frameFinal= imresize(frame1,[700 700]);
      writeVideo(video,frameFinal);
     videoPlayer(frameFinal);    
end

release(videoPlayer);
  
  close(video);
