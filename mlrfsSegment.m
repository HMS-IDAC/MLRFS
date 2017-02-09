% Multi-Layer Random Forest Segmentation: Segment

clear, clc

%% load image, model

Image = imread('~/Desktop/MLRFS/Test/I01.tif');
Label = imread('~/Desktop/MLRFS/Test/L01.tif'); % for comparison with rf segmentation output
rfModelFilePath = '/home/mc457/Desktop/MLRFS/Model/rfModel.mat';

disp('loading rfModel')
tic
load(rfModelFilePath); % loads rfModel
toc

%% segmentation

I0 = Image;
I = double(imresize(I0,rfModel.resizeFactor));
I = I/max(max(I));

% layer 1
F = imageFeatures(I,rfModel.sigmas);
fprintf('rf layer 1...'); tic
[imL,classProbs] = imclassify(F,rfModel.treeBags{1});
fprintf('time: %f s\n', toc);
    
% remaining layers
F = cat(3,F,repmat(zeros(size(I)),[1 1 rfModel.nLabels+rfModel.nProbMapsFeats]));
for layer = 2:rfModel.nLayers
    F(:,:,rfModel.nImageFeatures+1:rfModel.nImageFeatures+rfModel.nLabels) = classProbs;
    F(:,:,rfModel.nImageFeatures+rfModel.nLabels+1:end) = offsetFeatures(classProbs,rfModel.offsets);
    fprintf('rf layer %d...',layer); tic
    [imL,classProbs] = imclassify(F,rfModel.treeBags{layer});
    fprintf('time: %f s\n', toc);
end

%% display

L = imresize(imL,1/rfModel.resizeFactor,'nearest');
figureQSS
subplot(1,3,1)
imshow(I0), title('image')
subplot(1,3,2)
imshow(label2rgb(L)), title('rf segmentation output')
subplot(1,3,3)
imshow(label2rgb(Label)), title('ground truth')
