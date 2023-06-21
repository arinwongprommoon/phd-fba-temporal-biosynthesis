%% Dependencies
% Add the COBRA toolbox to MATLAB's search path
addpath(genpath('/home/arin/git/cobratoolbox'));

% Initialise COBRA toolbox
initCobraToolbox

%% load model

% Change the working directory
cd('/home/arin/git/fba-temporal-biosynthesis/');

% Specify the path to your SBML model in XML format
%sbmlPath = './models/yeast-GEM_8-6-0.xml';
sbmlPath = './models/ecYeastGEM_batch_8-6-0.xml';

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

disp(model_opt.f);
disp(sum(abs(model_opt.v)));

%% minimise taxicab norm
model_opt = optimizeCbModel(model1,'max','one');
v = model_opt.v;

disp(model_opt.f);
disp(model_opt.f1);
disp(sum(abs(model_opt.v)));

%% regularised FBA
model_opt = optimizeCbModel(model1,'max',1e-6);
v = model_opt.v;

%% (log) loop regularised FBA
reg_coeff_array = logspace(-12, -1, 200);
solutions_array = zeros(1, length(reg_coeff_array));
f2_array = zeros(1, length(reg_coeff_array));
sum_fluxes_array = zeros(1, length(reg_coeff_array));
for idx = 1:length(solutions_array)
    model_opt = optimizeCbModel(model1, 'max', reg_coeff_array(idx));
    solutions_array(idx) = model_opt.f;
    f2_array(idx) = model_opt.f2;
    sum_fluxes_array(idx) = sum(abs(model_opt.v));
end

%% (log) plots

tiledlayout(3,1);

% effect on growth rate
nexttile;
semilogx(reg_coeff_array, solutions_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel('Optimal growth rate [h^{-1}]');

% effect on second objective
% (squared Euclidean norm of internal fluxes)
nexttile;
loglog(reg_coeff_array, f2_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel({ ...
    'Value of second objective',
    '(squared Euclidean norm of internal fluxes' ...
    });

% effect on sum of fluxes
nexttile;
semilogx(reg_coeff_array, sum_fluxes_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel('Sum of absolute values of fluxes');

%% (lin) loop regularised FBA
reg_coeff_array = linspace(1.6e-7, 1.7e-7, 50);
solutions_array = zeros(1, length(reg_coeff_array));
f2_array = zeros(1, length(reg_coeff_array));
sum_fluxes_array = zeros(1, length(reg_coeff_array));
for idx = 1:length(solutions_array)
    model_opt = optimizeCbModel(model1, 'max', reg_coeff_array(idx));
    solutions_array(idx) = model_opt.f;
    f2_array(idx) = model_opt.f2;
    sum_fluxes_array(idx) = sum(abs(model_opt.v));
end

%% (lin) plots

tiledlayout(3,1);

% effect on growth rate
nexttile;
plot(reg_coeff_array, solutions_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel('Optimal growth rate [h^{-1}]');

% effect on second objective
% (squared Euclidean norm of internal fluxes)
nexttile;
plot(reg_coeff_array, f2_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel({ ...
    'Value of second objective',
    '(squared Euclidean norm of internal fluxes' ...
    });

% effect on sum of fluxes
nexttile;
plot(reg_coeff_array, sum_fluxes_array);
xlabel('Regularisation coefficient ($\sigma$)', 'Interpreter', 'latex');
ylabel('Sum of absolute values of fluxes');