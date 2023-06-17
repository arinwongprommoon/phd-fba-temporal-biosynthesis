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

model1 = model;
% glucose uptake
model1 = changeRxnBounds(model1, 'r_1714', -4.75, 'l')
% oxygen uptake
model1 = changeRxnBounds(model1, 'r_1992', -1000, 'l')

%% vanilla FBA
model_opt = optimizeCbModel(model1);
v = model_opt.v; 

%% minimise taxicab norm
model_opt = optimizeCbModel(model1,'max','one');
v = model_opt.v; 

%% regularised FBA
model_opt = optimizeCbModel(model1,'max',1e-6);
v = model_opt.v; 