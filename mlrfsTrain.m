% Multi-Layer Random Forest Segmentation: Train

clear, clc

%% set parameters

trainSetFolder = '~/Desktop/MLRFS/Train';
rfModelFolder = '~/Desktop/MLRFS/Model';

nLayers = 3;
% the algorithm will vertically split every training image in nLayers,
% so that every layer of the random forest sees a similar distribution of the training set;
% WARNING: every split of every training image should have representative pixels for
% all labels, otherwise the algorithm will crash

nLabels = 2; % labels are integers (1,2,...)
% see 'load training set' section below for how
% to load training images and labels

resizeFactor = 1/2;
% how much images are resized before training/testing
% there's an accuracy/speed trade-off in setting this value

% image features are simply derivatives (up to second order) in different scales;
% this parameter specifies such scales; details in imageFeatures.m
sigmas = [2 4 8];

offsets = [5 10]; % in pixels (after resizing); for offset features from probability maps (see offsetFeatures)

if nLayers < 2
    fprintf('nLayers should be at least 2\n');
    return
end
nImageFeatures = length(sigmas)*8; % see imageFeatures
nProbMapsFeats = length(offsets)*8*nLabels; % see offsetFeatures

%% load training set

% ----------------------------------------------------------------------
% NUMBER OF TRAINING IMAGES
% modify this parameter according to your training set
nTrainImages = 9;
% ----------------------------------------------------------------------

nImages0 = nTrainImages;
nImages = nLayers*nImages0;

imF = cell(1,nImages);
imL = cell(1,nImages);
for imIndex = 1:nImages0
    fprintf('features %d\n',imIndex);

    % ----------------------------------------------------------------------
    % LOAD IMAGE
    % modify this according to your training set
    I = imread(sprintf([trainSetFolder '/I%02d.tif'],imIndex));
    % ----------------------------------------------------------------------
    
    I = double(imresize(I,resizeFactor));
    I = I/max(max(I));

    [nr,nc] = size(I);
    
    % ----------------------------------------------------------------------
    % LOAD LABEL
    % modify this according to your training set
    L = imread(sprintf([trainSetFolder '/L%02d.tif'],imIndex));
    % ----------------------------------------------------------------------
    
    L = imresize(L,[nr,nc],'nearest');
    
%     imwrite(label2rgb(L,'jet','k'),'~/Desktop/Labels.png');
%     imshow(label2rgb(L,'jet','k')), pause
%     imtool(I), pause
    
    F = cat(3,imageFeatures(I,sigmas),repmat(zeros(size(I)),[1 1 nLabels+nProbMapsFeats]));
    nColsPerLayer = floor(nc/nLayers);
    for layer = 1:nLayers
        c0 = (layer-1)*nColsPerLayer;
        imF{(layer-1)*nImages0+imIndex} = F(:,c0+1:c0+nColsPerLayer,:);
        imL{(layer-1)*nImages0+imIndex} = L(:,c0+1:c0+nColsPerLayer,:);
    end
end

%% split training set

nImagesPerLayer = floor(nImages/nLayers);
layerF = cell(nLayers,nImagesPerLayer);
layerL = cell(nLayers,nImagesPerLayer);
for layer = 1:nLayers
    i0 = (layer-1)*nImagesPerLayer;
    for imIndex = 1:nImagesPerLayer
        layerF{layer,imIndex} = imF{i0+imIndex};
        layerL{layer,imIndex} = imL{i0+imIndex};
%         imshow(label2rgb(layerL{layer,imIndex},'jet','k'))
%         title(sprintf('layer %d, image %d', layer, imIndex))
%         pause
    end
end
clear imF
clear imL

%% train layer 1

ft = [];
lb = [];
for imIndex = 1:nImagesPerLayer
    F = layerF{1,imIndex};
    L = layerL{1,imIndex};
    [rfFeat,rfLbl] = rffeatandlab(F(:,:,1:nImageFeatures),L);
    ft = [ft; rfFeat];
    lb = [lb; rfLbl];
end
fprintf('training layer 1...'); tic
[treeBag,featImp] = train(ft,lb,20,60);
% figure, plot(featImp,'.'), title('feat imp 1')
fprintf('training time: %f s\n', toc);

save([rfModelFolder '/treeBag1.mat'],'treeBag');

%% train layers 2...nLayers

for layer = 2:nLayers
    ft = [];
    lb = [];
    for imIndex = 1:nImagesPerLayer
        F = layerF{layer,imIndex};
        L = layerL{layer,imIndex};

        for treeIndex = 1:layer-1
            load([rfModelFolder sprintf('/treeBag%d.mat',treeIndex)]);
            if treeIndex == 1
                [imL,classProbs] = imclassify(F(:,:,1:nImageFeatures),treeBag);
            else
                [imL,classProbs] = imclassify(F,treeBag);
            end
            F(:,:,nImageFeatures+1:nImageFeatures+nLabels) = classProbs;
            F(:,:,nImageFeatures+nLabels+1:end) = offsetFeatures(classProbs,offsets);
        end

        [rfFeat,rfLbl] = rffeatandlab(F,L);
        ft = [ft; rfFeat];
        lb = [lb; rfLbl];
    end

    fprintf('training layer %d...',layer); tic
    [treeBag,featImp] = train(ft,lb,20,60);
%     figure, plot(featImp,'.'), title(sprintf('feat imp %d',layer))
    fprintf('training time: %f s\n', toc);
    save([rfModelFolder sprintf('/treeBag%d.mat',layer)],'treeBag');
end

%% pack model

rfModel.nLayers = nLayers;
rfModel.nLabels = nLabels;
rfModel.resizeFactor = resizeFactor;
rfModel.sigmas = sigmas;
rfModel.offsets = offsets;
rfModel.nImageFeatures = nImageFeatures;
rfModel.nProbMapsFeats = nProbMapsFeats;
treeBags = cell(1,nLayers);
for i = 1:nLayers
    load([rfModelFolder sprintf('/treeBag%d.mat',i)]);
    treeBags{i} = treeBag;
end
rfModel.treeBags = treeBags;
disp('saving model')
tic
save([rfModelFolder '/rfModel.mat'],'rfModel','-v7.3');
toc
disp('done training')