function net = HardNetAffine(inputSize, hiddenSizes, outputSize)
    % Construct the MLP backbone used with the polygonal affine constraint layer.
    %% Arguments check
    if nargin < 1  %when the input size is not given
        inputSize = 4; 
    end
    if nargin < 2 %when the hidden sizes are not given
        hiddenSizes = [64 64]; 
    end
    if nargin < 3
        outputSize = 2; 
    end

    %% Construct layers for dlnetwork
    layers = [featureInputLayer(inputSize,'Normalization','none','Name','input')]; %normalization if expected from training script

    for k = 1:numel(hiddenSizes)
        layers = [layers; fullyConnectedLayer(hiddenSizes(k),'Name',sprintf('fc%d',k)); ...
            tanhLayer('Name',sprintf('tanh%d',k))]; %#ok<AGROW>
    end
    layers = [layers; fullyConnectedLayer(outputSize,'Name','force_raw')];

    %% Return the network
    net = dlnetwork(layers);
end
