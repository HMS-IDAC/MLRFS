function [treeBag,featImp] = train(rfFeat,rfLbl,ntrees,minleafsize)
% ntrees = 20; minleafsize = 60;

treeBag = TreeBagger(ntrees,rfFeat,rfLbl,'MinLeafSize',minleafsize,'oobvarimp','on');
featImp = treeBag.OOBPermutedVarDeltaError;

end