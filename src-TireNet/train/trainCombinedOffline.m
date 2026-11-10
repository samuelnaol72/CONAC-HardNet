function results = trainCombinedOffline(opt)
    % Train and compare models on one benchmark suite.
    %% Arguments Check
    arguments
        opt.DataFile (1,1) string = ""
        opt.Suite (1,1) string {mustBeMember(opt.Suite,["iid","sparse", ...
            "ood_slip","ood_environment","sparse_ood"])} = "iid"
        opt.OutputDir (1,1) string = fullfile(fileparts(mfilename('fullpath')),'results')
        opt.Epochs (1,1) double {mustBeInteger,mustBePositive} = 150
        opt.BatchSize (1,1) double {mustBeInteger,mustBePositive} = 256
        opt.LearnRate (1,1) double {mustBePositive} = 1e-3
        opt.HiddenSizes (1,:) double {mustBeInteger,mustBePositive} = [64 64]
        opt.Methods (1,:) string {mustBeMember(opt.Methods,["raw","affine","plusplus"])} = ...
            ["raw","affine","plusplus"]
        opt.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 42
        opt.SparseFraction (1,1) double {mustBeGreaterThan(opt.SparseFraction,0), ...
            mustBeLessThanOrEqual(opt.SparseFraction,1)} = 0.10
        opt.MaxIterations (1,1) double {mustBeInteger,mustBePositive} = 40
        opt.Damping (1,1) double {mustBePositive} = 1e-8
        opt.Tolerance (1,1) double {mustBePositive} = 1e-4
        opt.PolygonSides (1,1) double {mustBeInteger, ...
            mustBeGreaterThanOrEqual(opt.PolygonSides,8)} = 32
        opt.ConstraintWeight (1,1) double {mustBeNonnegative} = 0
        opt.Regularization (1,1) double {mustBeNonnegative} = 1e-7
    end

    %% Prepare Data and Network Settings
    here=fileparts(mfilename('fullpath'));
    addpath(fullfile(here,'..','data'));
    addpath(fullfile(here,'..','models'));
    args={'Suite',opt.Suite,'Seed',opt.Seed, ...
        'SparseFraction',opt.SparseFraction};
    if strlength(opt.DataFile)>0
        args=[args,{'File',opt.DataFile}];
    end
    d=prepareCombinedSlip(args{:});
    if ~isfolder(opt.OutputDir), mkdir(opt.OutputDir); end

    % Use train-only normalization; DPL damping matches force units in N.
    yMu=[d.normalization.target.longitudinal_force.mean; ...
        d.normalization.target.lateral_force.mean];
    ySd=[d.normalization.target.longitudinal_force.std; ...
        d.normalization.target.lateral_force.std];
    yMu=dlarray(single(yMu)); ySd=dlarray(single(ySd));
    tr=pack(d.train); va=pack(d.validation); te=pack(d.test);
    cfg=struct('MaxIterations',opt.MaxIterations, ...
        'Damping',opt.Damping,'Tolerance',opt.Tolerance, ...
        'PolygonSides',opt.PolygonSides, ...
        'ConstraintWeight',opt.ConstraintWeight, ...
        'Regularization',opt.Regularization, ...
        'targetMean',yMu,'targetStd',ySd);

    %% Train Baselines and Hard-Constrained Models
    methods=opt.Methods;
    results=struct();
    for m=1:numel(methods)
        rng(opt.Seed,'twister');
        method=methods(m); cfg.Method=char(method);
        if method=="plusplus"
            net=HardNetPlusPlus(4,opt.HiddenSizes,2);
        else
            net=HardNetAffine(4,opt.HiddenSizes,2);
        end
        av=[]; av2=[]; iter=0;
        n=size(tr.X,2); hist=zeros(opt.Epochs,4);

        for epoch=1:opt.Epochs
            order=randperm(n); 
            bLoss=0; nBatch=0;
            for first=1:opt.BatchSize:n
                ids=order(first:min(first+opt.BatchSize-1,n));
                [loss,grad]=dlfeval(@combinedForceLoss,net, ...
                    tr.X(:,ids),tr.Y(:,ids),tr.limits(:,ids),cfg);
                iter=iter+1;
                [net,av,av2]=adamupdate(net,grad,av,av2,iter,opt.LearnRate);
                bLoss=bLoss+double(gather(extractdata(loss)));
                nBatch=nBatch+1;
            end

            vaStats=evaluate(net,va,cfg);
            hist(epoch,:)=[bLoss/nBatch,vaStats.elementRMSE, ...
                vaStats.maxResidual,vaStats.meanIterations];
            if epoch==1 || mod(epoch,10)==0 || epoch==opt.Epochs
                fprintf('%s/%s epoch %d/%d train %.5g val RMSE %.2f N\n', ...
                    opt.Suite,method,epoch,opt.Epochs,hist(epoch,1), ...
                    vaStats.elementRMSE);
            end
        end

        results.(char(method)).net=net;
        results.(char(method)).history=array2table(hist,'VariableNames', ...
            {'trainLoss','validationElementRMSE_N', ...
            'validationMaxResidual','validationMeanIterations'});
        results.(char(method)).validation=evaluate(net,va,cfg);
        results.(char(method)).test=evaluate(net,te,cfg);
        results.(char(method)).config=cfg;
        fprintf('%s/%s test RMSE %.2f N; tolerance exceedance %.2f%%\n', ...
            opt.Suite,method,results.(char(method)).test.elementRMSE, ...
            results.(char(method)).test.toleranceViolationRate);
    end

    %% Save Results
    results.suite=opt.Suite;
    results.split=d.split;
    results.normalization=d.normalization;
    results.envelope=d.envelope;
    results.datasetInfo=struct('source',d.source, ...
        'nTrain',size(d.train.X,1), ...
        'nValidation',size(d.validation.X,1), ...
        'nTest',size(d.test.X,1), ...
        'seed',opt.Seed,'sparseFraction',opt.SparseFraction);
    outFile=fullfile(opt.OutputDir,char(opt.Suite+"_models.mat"));
    save(outFile,'results','-v7.3');

    function p=pack(s)
        p.X=dlarray(single(s.X'),'CB');
        p.Y=dlarray(single(s.Y'),'CB');
        p.limits=dlarray(single(s.limits'),'CB');
        p.truth=s.Yraw;
    end
end

function stats=evaluate(net,d,cfg)
    [~,~,s]=dlfeval(@combinedForceLoss,net,d.X,d.Y,d.limits,cfg);
    pred=gather(extractdata(s.predictionPhysical))';
    err=pred-d.truth;
    y=s.predictionPhysical; lim=d.limits;
    res=gather(extractdata(max((y(1,:)./lim(1,:)).^2+ ...
        (y(2,:)./lim(2,:)).^2-1,0)))';

    stats.prediction=pred;
    stats.componentRMSE=sqrt(mean(err.^2,1));
    stats.elementRMSE=sqrt(mean(err.^2,'all'));
    stats.ellipseResidual=res;
    stats.violationRate=100*mean(res>0);
    stats.toleranceViolationRate=100*mean(res>cfg.Tolerance);
    stats.maxResidual=max(res,[],'all');
    stats.meanIterations=double(s.meanIterations);
end
