function [loss,gradients,stats] = combinedForceLoss(net,X,Y,limits,opt)
    % X is 4xB; Y and limits are 2xB, in CB data format.
    %% Forward Pass and Output Constraint
    raw=forward(net,X);
    rawPhys=raw.*opt.targetStd+opt.targetMean;
    switch lower(string(opt.Method))
        case "plusplus"
            [predPhys,res,count]=ellipseProjection(rawPhys,limits,opt);
        case "affine"
            [predPhys,res,count]=polygonProjection(rawPhys,limits,opt);
        case "raw"
            predPhys=rawPhys;
            res=max((rawPhys(1,:)./limits(1,:)).^2+ ...
                (rawPhys(2,:)./limits(2,:)).^2-1,0);
            count=zeros(1,size(raw,2),'like',extractdata(raw));
        otherwise
            error('Unknown constraint method: %s',opt.Method);
    end

    %% Calculate Loss and Gradients
    pred=(predPhys-opt.targetMean)./opt.targetStd;
    fitLoss=mean(sum((pred-Y).^2,1),'all');
    cstrLoss=mean(res.^2,'all');
    regLoss=0;
    for k=1:height(net.Learnables)
        regLoss=regLoss+sum(net.Learnables.Value{k}.^2,'all');
    end
    loss=fitLoss+opt.ConstraintWeight*cstrLoss+ ...
        opt.Regularization*regLoss;
    gradients=dlgradient(loss,net.Learnables);
    stats=struct('fitLoss',fitLoss,'constraintLoss',cstrLoss, ...
        'maxResidual',max(res,[],'all'), ...
        'meanIterations',mean(count,'all'),'prediction',pred, ...
        'rawPrediction',raw,'predictionPhysical',predPhys, ...
        'rawPredictionPhysical',rawPhys);
end

function [y,res,count]=ellipseProjection(y,limits,opt)
    % Damped HardNet++ correction for the friction ellipse.
    count=zeros(1,size(y,2),'like',extractdata(y));
    for k=1:opt.MaxIterations
        c=(y(1,:)./limits(1,:)).^2+ ...
            (y(2,:)./limits(2,:)).^2-1;
        active=c>opt.Tolerance;
        if ~any(extractdata(active),'all'), break; end
        gx=2*y(1,:)./limits(1,:).^2;
        gy=2*y(2,:)./limits(2,:).^2;
        step=max(c,0)./(gx.^2+gy.^2+opt.Damping);
        y=y-[gx.*step;gy.*step];
        count=count+double(extractdata(active));
    end
    res=max((y(1,:)./limits(1,:)).^2+ ...
        (y(2,:)./limits(2,:)).^2-1,0);
end

function [y,res,count]=polygonProjection(y,limits,opt)
    % Project against the affine half-spaces of a circumscribed polygon.
    n=opt.PolygonSides;
    theta=(0:n-1)'*(2*pi/n);
    normal=[cos(theta),sin(theta)];
    count=zeros(1,size(y,2),'like',extractdata(y));

    for k=1:opt.MaxIterations
        for j=1:n
            a1=normal(j,1)./limits(1,:);
            a2=normal(j,2)./limits(2,:);
            c=a1.*y(1,:)+a2.*y(2,:)-1;
            active=c>opt.Tolerance;
            if any(extractdata(active),'all')
                step=max(c,0)./(a1.^2+a2.^2+opt.Damping);
                y=y-[a1.*step;a2.*step];
                count=count+double(extractdata(active));
            end
        end
        z1=y(1,:)./limits(1,:);
        z2=y(2,:)./limits(2,:);
        c=normal(:,1).*z1+normal(:,2).*z2-1;
        if ~any(extractdata(max(c,[],1)>opt.Tolerance),'all'), break; end
    end

    z1=y(1,:)./limits(1,:);
    z2=y(2,:)./limits(2,:);
    res=max(max(normal(:,1).*z1+normal(:,2).*z2-1,[],1),0);
end
