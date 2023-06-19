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

%% loop regularised FBA
% TODO: test code
reg_coeff_array = logspace(1e-6, 1e-1, 10);
solutions_array = zeros(1, length(reg_coeff_array));
sum_fluxes_array = zeros(1, length(reg_coeff_array));
for idx = 1:length(solutions_array)
    model_opt = optimizeCbModel(model1, 'max', reg_coeff_array(idx));
    solutions_array(idx) = model_opt.f;
    sum_fluxes_array(idx) = sum(model_opt.v);
end

%% plot effect on growth rate
% TODO: test code
semilogx(reg_coeff_array, solutions_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel('Optimal growth rate (h^{-1})');

%% plot effect on sum of fluxes
% TODO: test code
semilogx(reg_coeff_array, solutions_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel('Optimal growth rate (h^{-1})');
