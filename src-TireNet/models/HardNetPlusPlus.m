function net = HardNetPlusPlus(inputSize, hiddenSizes, outputSize)
    % Construct the force-prediction backbone used with the nonlinear ellipse DPL.

    %%  Arguments check
    if nargin < 1
        inputSize = 4; 
    end
    if nargin < 2
        hiddenSizes = [64 64]; 
    end
    if nargin < 3
        outputSize = 2; 
    end

    %% Construct layer graph for dlnetwork
    layers = [featureInputLayer(inputSize,'Normalization','none','Name','input')];  %normalization if expected from training script
    for k = 1:numel(hiddenSizes)
        layers = [layers; fullyConnectedLayer(hiddenSizes(k),'Name',sprintf('fc%d',k)); ...
            tanhLayer('Name',sprintf('tanh%d',k))]; %#ok<AGROW>
    end
    layers = [layers; fullyConnectedLayer(outputSize,'Name','force_raw')];

     %% Return the network
    net = dlnetwork(layers);
end
