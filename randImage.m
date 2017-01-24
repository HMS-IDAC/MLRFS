function [I,L] = randImage(nRows,nCols,nCircles,radiiRange)
    [x,y] = meshgrid(1:nCols,1:nRows);
    I = 0.5*rand(nRows,nCols);
    L = ones(nRows,nCols);
    for cIndex = 1:nCircles
        x0 = randi(nCols);
        y0 = randi(nRows);
        r = radiiRange(1)+rand*diff(radiiRange);
        d = sqrt((x-x0).^2+(y-y0).^2);
        idx = reshape(d < r, 1,[]);
        I(idx) = rand;
        L(idx) = 2;
    end
end