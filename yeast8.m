%% Dependencies
% Add the COBRA toolbox to MATLAB's search path
addpath(genpath('/home/arin/git/cobratoolbox'));

% Initialise COBRA toolbox
initCobraToolbox

%% load model

% Change the working directory
cd('/home/arin/git/fba-temporal-biosynthesis/');

% Specify the path to your SBML model in XML format
sbmlPath = './models/yeast-GEM_8-6-0.xml';

% Read the SBML model using the COBRA toolbox
model = readCbModel(sbmlPath);

%% modify model

% change reaction bounds where necessary
% this code is copied from Denise and hasn't been adapted to ecYeast8.6.0
% FIXME: adapt
model1 = model;
model1 = changeRxnBounds(model1,model.rxns(687),0.005,'b'); %D-Lactate release
model1 = changeRxnBounds(model1,model.rxns(3446),-10,'b'); %O2 uptake

%% vanilla FBA
model_opt = optimizeCbModel(model);
v = model_opt.v; 

%% minimise taxicab norm
model_opt = optimizeCbModel(model,'max','one');
v = model_opt.v; 

%% regularised FBA
model_opt = optimizeCbModel(model,'max',1e-6);
v = model_opt.v; 