% Generates synthetic data for training/testing

trainPath = '~/Desktop/MLRFS/Train';
testPath = '~/Desktop/MLRFS/Test';

nTrain = 9;
nTest = 1;

nRows = 400;
nCols = 600;

radiiRange = [10 20];

nCircles = 20;

for imIndex = 1:nTrain
    [I,L] = randImage(nRows,nCols,nCircles,radiiRange);
%     imshow([I (L == 1) (L == 2)]), pause
    imwrite(uint8(255*I),sprintf([trainPath '/I%02d.tif'],imIndex));
    imwrite(uint8(L),sprintf([trainPath '/L%02d.tif'],imIndex));
end

for imIndex = 1:nTest
    [I,L] = randImage(nRows,nCols,nCircles,radiiRange);
    imwrite(uint8(255*I),sprintf([testPath '/I%02d.tif'],imIndex));
    imwrite(uint8(L),sprintf([testPath '/L%02d.tif'],imIndex));
end