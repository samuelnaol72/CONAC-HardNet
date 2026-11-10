function data = prepareCombinedSlip(opt)
    % Build reproducible, suite-specific tire-force datasets.
    %% Arguments Check
    arguments
        opt.File (1,1) string = ""
        opt.Suite (1,1) string {mustBeMember(opt.Suite,["iid","sparse", ...
            "ood_slip","ood_environment","sparse_ood","all"])} = "iid"
        opt.Seed (1,1) double {mustBeInteger,mustBeNonnegative} = 42
        opt.SparseFraction (1,1) double {mustBeGreaterThan(opt.SparseFraction,0), ...
            mustBeLessThanOrEqual(opt.SparseFraction,1)} = 0.10
        opt.ForceThresholdkN (1,1) double {mustBePositive} = 50
        opt.EllipseGamma (1,1) double {mustBePositive} = 1.012
        opt.RawOutputRoot (1,1) string = ""
        opt.ProcessedOutputRoot (1,1) string = ""
        opt.Export (1,1) logical = true
    end

    %% Load and Clean the Dataset
    if strlength(opt.File)==0
        opt.File=fullfile(fileparts(mfilename('fullpath')),'tire_rig_dataset.csv');
    end
    if ~isfile(opt.File)
        error('Dataset file not found: %s',opt.File); 
    end
    if strlength(opt.RawOutputRoot)==0
        opt.RawOutputRoot=fullfile(fileparts(mfilename('fullpath')),'raw_suites');
    end
    if strlength(opt.ProcessedOutputRoot)==0
        opt.ProcessedOutputRoot=fullfile(fileparts(mfilename('fullpath')),'processed_suites');
    end

    T=readtable(opt.File,'VariableNamingRule','preserve');  % read raw table
    xNames=["slip_ratio","slip_angle",...                   % input features
    "normal_load","friction_coefficient"];  
    yNames=["longitudinal_force","lateral_force"];          % network output
    required=[xNames,yNames,"env_id","tire_id"];
    missing=setdiff(required,string(T.Properties.VariableNames));
    if ~isempty(missing)                                    % check missing columns
        error('Missing CSV columns: %s',strjoin(missing,', '));
    end

    numNames=[xNames,yNames];
    for k=1:numel(numNames)                                 % incase the values are in string ex) "0.15" >> 0.15
        if ~isnumeric(T.(numNames(k)))
            T.(numNames(k))=str2double(string(T.(numNames(k))));
        end
    end

    % keep where numNames has a finite numeric value.
    mask=all(isfinite(T{:,numNames}),2);
    T=T(mask,:);
    fLimit=opt.ForceThresholdkN*1000;
    T=T(abs(T.longitudinal_force)<=fLimit & ...
        abs(T.lateral_force)<=fLimit,:);
    if isempty(T)
        error('No valid samples remain after cleaning.'); 
    end

    %% Prepare Selected Benchmark Suite(s)
    % List the supported benchmark suites and select the requested suite(s).
    % "all" runs the same preprocessing steps independently for every suite.
    suiteNames=["iid","sparse","ood_slip","ood_environment","sparse_ood"];
    suiteList=suiteNames;
    if opt.Suite~="all" 
        suiteList=opt.Suite; 
    end

    % Store each prepared suite under its name in this output struct.
    suiteSet=struct();

    for k=1:numel(suiteList)
        name=char(suiteList(k));

        % Get row indices for this suite's train, validation, and test splits.
        [iTr,iVa,iTe,splitDef]=makeSplit(T,suiteList(k), ...
            opt.Seed,opt.SparseFraction);

        % Normalize using training rows, attach labels and envelope limits,
        % and keep the split definition and source rows as metadata.
        suiteSet.(name)=packageSuite(T,iTr,iVa,iTe,splitDef,suiteList(k));

        % Optionally write the raw CSVs, processed MAT files, and metadata.
        if opt.Export
            exportSuite(suiteSet.(name),opt);
        end

        % Report the number of rows assigned to each split.
        fprintf('%-16s train=%5d valid=%5d test=%5d\n',name, ...
            height(suiteSet.(name).train.rows), ...
            height(suiteSet.(name).validation.rows), ...
            height(suiteSet.(name).test.rows));
    end

    % Return either the requested suite or the struct containing all suites.
    if opt.Suite=="all"
        data=suiteSet;
    else
        data=suiteSet.(char(opt.Suite));
    end

    function suite=packageSuite(T0,iTr,iVa,iTe,splitDef,suiteName)
        Ttr=T0(iTr,:); Tva=T0(iVa,:); Tte=T0(iTe,:);
        normStats=normalizationStats(Ttr,xNames,yNames);
        suite=struct('name',suiteName,'source',opt.File, ...
            'train',makePartition(Ttr,normStats,xNames,yNames,opt.EllipseGamma), ...
            'validation',makePartition(Tva,normStats,xNames,yNames,opt.EllipseGamma), ...
            'test',makePartition(Tte,normStats,xNames,yNames,opt.EllipseGamma), ...
            'normalization',normStats,'featureNames',xNames, ...
            'targetNames',yNames,'contextNames', ...
            ["normal_load","friction_coefficient"], ...
            'envelope',makeEnvelopeMetadata(opt.EllipseGamma));
        suite.split=struct('mode',splitDef.type,'counts',struct( ...
            'train',height(Ttr),'valid',height(Tva),'test',height(Tte)), ...
            'definition',splitDef, ...
            'trainEnvironments',{uniqueStrings(Ttr.env_id)}, ...
            'validationEnvironments',{uniqueStrings(Tva.env_id)}, ...
            'testEnvironments',{uniqueStrings(Tte.env_id)});
    end
end

function [iTr,iVa,iTe,info]=makeSplit(T,suite,seed,sparseFrac)
    % Returns the suite specific indices from table T
    n=height(T); 
    allIdx=(1:n)';

    %% IID and Sparse Random-Row Suites
    if suite=="iid" || suite=="sparse"
        rng(seed,'twister'); 
        order=allIdx(randperm(n));  % create random ordering of integers 1 through n
        nTr=floor(.70*n);   %  70% of the dataset is training dataset
        nVa=floor(.15*n);
        iTr=order(1:nTr);
        iVa=order(nTr+1:nTr+nVa);
        iTe=order(nTr+nVa+1:end);
        if suite=="sparse"
            iTr=sparseSample(T,iTr,sparseFrac,seed+1);  % make training indices sparse
            frac=sparseFrac;
        else
            frac=1;
        end
        info=struct('type','random_rows', ...
            'train_fraction_of_eligible_rows',.70*frac, ...
            'sparse_fraction',frac);
        return
    end

    %% Slip-OOD Suite
    slipOOD=abs(T.slip_ratio)>.10 | rad2deg(abs(T.slip_angle))>6; % test indices
    if suite=="ood_slip"
        [iTr,iVa]=randomSplit(find(~slipOOD),.80,seed);
        iTe=find(slipOOD);
        info=struct('type','slip_domain_holdout', ...
            'training_region','abs(slip_ratio)<=0.10 and abs(slip_angle)<=6 deg', ...
            'test_region','complement of training region');
        return
    end

    %% Environment-OOD and Combined Sparse-OOD Suites
    [trAll,vaAll,teAll,envInfo]=strictContextSplit(T,seed);
    if suite=="ood_environment"
        iTr=trAll; iVa=vaAll; iTe=teAll; info=envInfo;
        return
    end
    trCand=trAll(~slipOOD(trAll));
    iVa=vaAll(~slipOOD(vaAll));
    iTr=sparseSample(T,trCand,sparseFrac,seed+1);
    iTe=unique([find(slipOOD);teAll]);
    info=struct('type','sparse_union_ood_holdout', ...
        'training_region','slip and load/friction context in-domain', ...
        'test_region','slip OOD union strict load/friction factor holdout', ...
        'sparse_fraction',sparseFrac,'strict_context_holdout',envInfo);
end

function [iTr,iVa]=randomSplit(indices,fraction,seed)
    % shuffles a list of elligble row indices and then divides
    % into training and validation indices
    rng(seed,'twister'); 
    indices=indices(randperm(numel(indices))); % shuffling
    if numel(indices)<2
        error('At least two eligible samples are needed for a split.');
    end
    nTr=max(1,round(fraction*numel(indices)));
    nTr=min(nTr,numel(indices)-1);
    iTr=indices(1:nTr); 
    iVa=indices(nTr+1:end);
end

function iOut=sparseSample(T,indices,fraction,seed)
    % Sparse sampling within each environment.
    if ~(fraction>0 && fraction<=1)
        error('SparseFraction must be in (0,1].');
    end

    % Read the environment label for each eligible row and list unique labels.
    env=string(T.env_id(indices));
    groups=unique(env,'sorted'); % Unique environments among eligible rows.
    iOut=[];
    rng(seed,'twister');

    % Sample each environment separately so sparse selection preserves coverage.
    for k=1:numel(groups)
        msk=env==groups(k); % True for rows in this environment.
        rows=indices(msk);  % Original row indices in T.
        nRows=numel(rows);
        nTake=min(nRows,max(1,round(fraction*nRows)));
        iRand=randperm(nRows,nTake);  %return random position for rows
        iOut=[iOut;rows(iRand)]; %#ok<AGROW>
    end

    % Sort selected indices for stable row order in the exported dataset.
    iOut=sort(iOut);
end

function [iTr,iVa,iTe,info]=strictContextSplit(T,seed)
    loads=unique(T.normal_load); mus=unique(T.friction_coefficient);
    if numel(loads)<4 || numel(mus)<4
        error('Environment-OOD requires at least four load and friction levels.');
    end
    testLoads=[min(loads),max(loads)]; testMus=[min(mus),max(mus)];
    loadMask=ismember(T.normal_load,testLoads);
    muMask=ismember(T.friction_coefficient,testMus);
    testMask=loadMask & muMask;
    eligible=find(~loadMask & ~muMask);
    [iTr,iVa]=randomSplit(eligible,.80,seed);
    iTe=find(testMask);
    info=struct('type','strict_factor_holdout', ...
        'training_region','load and friction levels exclude both test factor sets', ...
        'test_region','held-out load levels AND held-out friction levels', ...
        'test_load_levels_N',testLoads,'test_friction_levels',testMus, ...
        'excluded_rows',height(T)-numel(iTr)-numel(iVa)-numel(iTe));
end

function stats=normalizationStats(T,xNames,yNames)
    stats=struct('features',struct(),'target',struct());
    for k=1:numel(xNames)
        col=char(xNames(k)); x=T.(col); s=std(x,0);
        if ~isfinite(s) || s<1e-12, s=1; end
        stats.features.(col)=struct('mean',mean(x),'std',s);
    end
    for k=1:numel(yNames)
        col=char(yNames(k)); y=T.(col); s=std(y,0);
        if ~isfinite(s) || s<1e-12, s=1; end
        stats.target.(col)=struct('mean',mean(y),'std',s);
    end
end

function part=makePartition(T,normStats,xNames,yNames,gamma)
    xRaw=T{:,xNames}; yRaw=T{:,yNames};
    x=xRaw; y=yRaw;
    for k=1:numel(xNames)
        st=normStats.features.(char(xNames(k)));
        x(:,k)=(xRaw(:,k)-st.mean)/st.std;
    end
    for k=1:numel(yNames)
        st=normStats.target.(char(yNames(k)));
        y(:,k)=(yRaw(:,k)-st.mean)/st.std;
    end
    if ismember('test_family',T.Properties.VariableNames)
        family=string(T.test_family);
    else
        family=repmat("unknown",height(T),1);
    end

    %% Fixed Load-Dependent Friction Ellipse
    fz=T.normal_load; mu=T.friction_coefficient; pn=3580.65;
    sx=1.461-0.0495*(fz/pn);
    sy=1.3601625-0.1090375*(fz/pn);
    limits=[gamma.*sx.*mu.*fz,gamma.*sy.*mu.*fz];
    part=struct('X',x,'Xraw',xRaw,'Y',y,'Yraw',yRaw, ...
        'limits',limits,'contextRaw',T{:,{'normal_load','friction_coefficient'}}, ...
        'envId',string(T.env_id),'testFamily',family,'rows',T);
end

function metadata=makeEnvelopeMetadata(gamma)
    metadata=struct('type','load_dependent_affine_ellipse_polygon', ...
        'pn_N',3580.65,'gamma',gamma,'polygon_sides',32,'halfspace_rhs',1, ...
        'scale_x','1.461 - 0.0495 * (Fz / pn_N)', ...
        'scale_y','1.3601625 - 0.1090375 * (Fz / pn_N)', ...
        'Fx_limit','gamma * scale_x(Fz) * mu * Fz', ...
        'Fy_limit','gamma * scale_y(Fz) * mu * Fz', ...
        'constraint','(Fx/Fx_limit)^2 + (Fy/Fy_limit)^2 <= 1', ...
        'calibration','fixed model-level constants; no dataset fit');
end

function exportSuite(suite,opt)
    rawDir=fullfile(opt.RawOutputRoot,char(suite.name));
    procDir=fullfile(opt.ProcessedOutputRoot,char(suite.name));
    if ~isfolder(rawDir), mkdir(rawDir); end
    if ~isfolder(procDir), mkdir(procDir); end
    parts={'train','validation','test'}; names={'train','valid','test'};
    for k=1:3
        part=suite.(parts{k});
        writetable(part.rows,fullfile(rawDir,[names{k} '.csv']));
        save(fullfile(procDir,[names{k} '.mat']),'-struct','part');
    end
    counts=suite.split.counts;
    meta=struct('preprocessor','prepareCombinedSlip.m', ...
        'source_dataset_file',suite.source,'suite',suite.name, ...
        'total_source_samples',counts.train+counts.valid+counts.test, ...
        'split',suite.split,'feature_names',suite.featureNames, ...
        'target_names',suite.targetNames,'context_names',suite.contextNames, ...
        'normalization_stats',suite.normalization, ...
        'hardnet_constraint',suite.envelope, ...
        'raw_label_policy','Raw force labels are preserved.');
    writeJson(fullfile(procDir,'metadata.json'),meta);
    writeJson(fullfile(rawDir,'manifest.json'),suite.split);
end

function writeJson(path,value)
    fid=fopen(path,'w');
    if fid<0, error('Could not write JSON: %s',path); end
    cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fwrite(fid,jsonencode(value,'PrettyPrint',true),'char');
end

function envs=uniqueStrings(values)
    envs=cellstr(sort(unique(string(values))));
end
