% Frank Longueira & Matthew Smarsch
% Stochastic Processes - MATLAB Project 1
% Prof. Keene
% 4/4/15

clear all;
close all;
clc;

%% Scenario 1

% (a) - Bayesian MMSE Estimator

iter = 1000; % Number of iterations
num =  100;  % Number of observations

h = 0.5;     % This is the h as stated in Scenario 1
mu_O = 7;   % Mean of Theta
var_O = 4;   % Variance of Theta
var_Noise = 8; % Variance of the Noise
tau = 1/var_Noise;
tau_O = 1/var_O;

mu_O_est = zeros(1,num);     % Pre-allocation of memory
MSE_num = zeros(1,num);      % Pre-allocation of memory
MSE_iter = zeros(iter,num);  % Pre-allocation of memory
MSE_Distrib = zeros(3,num);  % Pre-allocation of memory

for j = 1:3 % Chooses which distribution Theta comes from
    for n = 1:iter
        x = zeros(1,num);
        Distrib = [normrnd(mu_O,sqrt(var_O)) 2*mu_O*rand exprnd(1/mu_O)];
        Theta = Distrib(j);
        for k = 1:num
            x(k)= h*Theta + normrnd(0,sqrt(var_Noise)); % Adds noise to Theta
            x_bar = sum(x)/k; % Finds running sample mean
            mu_O_est(k) = (k*tau*x_bar+tau_O*mu_O)/((k*tau+tau_O)*h); % This is the Baye's MMSE Estimate for Theta that gets better as the number of observations increases
            MSE_num(k) = (Theta - mu_O_est(k))^2; % This is the MSE for Baye's MMSE estimate for each number of observations
        end
        MSE_iter(n,:) = MSE_num;
    end
MSE_Distrib(j,:) = mean(MSE_iter); % This is the averaged MSE data over the iterations, for each distribution, as a function of the number of observations.
end


% (b) - ML Estimator

mu_num_O = zeros(1,num);   % Pre-allocation of memory
mu_iter = zeros(iter,num); % Pre-allocation of memory
MSE_num = zeros(1,num);    % Pre-allocation of memory
MSE_iter = zeros(iter,num);% Pre-allocation of memory

Theta = mu_O;   % Fixed Theta for ML Estimate
var_Noise = 12; % Variance of the added noise


for n = 1:iter
    x = zeros(1,num);
    for k = 1:num
        x(k) = h*Theta + normrnd(0,sqrt(var_Noise)); % Adds noise to each observation of the fixed theta
        x_bar = sum(x)/(k); % This is the running sample mean of X
        mu_num_O(k) = x_bar/h; % This is the estimate of Theta that gets better per observation
        MSE_num(k)= ((Theta)- mu_num_O(k))^2 ; % This is the MSE for ML estimate for each number of observations
    end
    mu_iter(n,:) = mu_num_O;
    MSE_iter(n,:) = MSE_num;
end


MSE_ML = mean(MSE_iter); % This is the averaged MSE data over the iterations as a function of the number of observations.

% (c)  Plotting MSE for each estimation:
figure;
plot(1:num,MSE_Distrib(1,:),'b',1:num,MSE_ML,'r')
title('Bayes MMSE vs. ML Estimate')
ylabel('MSE')
xlabel('Number of Observations')
ylim([0 20])
legend('Gaussian Prior - Bayes MMSE', 'ML Estimate')

figure;
plot(1:num,MSE_Distrib(1,:),'b',1:num,MSE_Distrib(2,:),'g',1:num,MSE_Distrib(3,:), 'k')
title('Incorrect Prior Knowledge for Bayes MMSE')
ylabel('MSE')
xlabel('Number of Observations')
ylim([0 20])
legend('Gaussian Prior', 'Uniform Prior','Exponential Prior')



%% Scenario 3
%% Mock simulation to check part(d)'s theoretical results
trials = 3e5;
X1 = randi(2,trials,2);
PY_2 = sum(X1(:,1)+X1(:,2)==2)/trials; % Probability of FIREBALL damage = 2
PY_3 = sum((X1(:,1)+X1(:,2)==3))/trials; % Probability of FIREBALL damage = 2

Trolls = randi(4,trials,6); % Generating 6 uniformly chosen HP's for each troll for each trial

bb = zeros(2,3); % Pre-allocating memory

for k=1:2
    Y= (k+1)*ones(1,6); % Creating a FIREBALL Matrix that is the same size as the Trolls matrix generated above
    bucket = 0;
    bucket2 = 0;
    bucket3 = 0;
    for n = 1:trials
        if sum(Trolls(n,:)>Y) == 1; % Look for trials in which exactly 5 trolls were slayed
            bucket = bucket + 1;
            if length(find(Trolls(n,:)-Y==1))==1; % Look for trials where the 1 troll remaining has HP = 1
                bucket2 = bucket2 + 1; 
            elseif length(find(Trolls(n,:)-Y==2))==1; % Look for trials where the 1 troll remaining has HP = 2
                bucket3 = bucket3 + 1;
            end
        bb(k,1:3) = [bucket bucket2 bucket3];
        end
    end
end
P_5_slayed = (bb(1,1)*PY_2 + bb(2,1)*PY_3)/trials; % Probability of 5 trolls slayed
P_5_slayed_Z1 = (bb(1,2)*PY_2+bb(2,2)*PY_3)/trials; % Probability of (5 trolls Slayed & HP = 1)
P_5_slayed_Z2 = (bb(1,3)*PY_2+bb(2,3)*PY_3)/trials; % Probability of (5 trolls slayed & HP = 2)

P_Z1 = P_5_slayed_Z1/P_5_slayed; % Probability of HP=1 given 5 trolls were slayed
P_Z2 = P_5_slayed_Z2/P_5_slayed; % Probability of HP=2 given 5 trolls were slayed

Expected_HP_Remaining = (1)*P_Z1+(2)*P_Z2 % Expected value of last troll's HP


%% (c) - Simiulating Part (d) for Arbitrary Level


Level = 10;     % Level of trolls/Keene

trials = 3e5;   % Number of experiments performed
X1 = randi(Level,trials,2); % Generating trials for 2 rolls of "Level"-sided dice
P_Y = zeros(1,2*Level); % Pre-allocation

for jj = 2:2*Level
    P_Y(jj)= sum(X1(:,1)+X1(:,2)==jj)/trials; % P_Y(jj) is the probability of FIREBALL damage = jj
end

Num_Trolls = 6;
Trolls = randi(2*Level,trials,Num_Trolls); % Generating 6 uniformly chosen HP's for each troll for each trial



bb = zeros(2*Level,2*Level);          % Pre-allocating memory
P_5_trolls_slayed = zeros(1,2*Level); % Pre-allocating memory

for k=1:2*Level
    Y= (k)*ones(1,Num_Trolls); % Creating a FIREBALL Matrix that is the same size as the Trolls matrix generated above
    bucket_5trolls = 0;
    for n = 1:trials
        if sum(Trolls(n,:)>Y) == 1; % Look for trials in which exactly 5 trolls were slayed
            bucket_5trolls = bucket_5trolls + 1;
            for hh = 1:2*Level
                if length(find(Trolls(n,:)-Y==hh))==1; % Look for trials where the troll remaining has HP = hh
                    bb(k,hh) = bb(k,hh) + 1; 
                end
            end
        end
    P_5_trolls_slayed(k) = bucket_5trolls;
    end
end
P_5trolls_slayed = sum((P_5_trolls_slayed/trials).*P_Y); % Probability that 5 trolls are slayed out of 6

P_ZN_5trolls = zeros(1,2*Level); % Pre-allocation

for uu=1:2*Level
    P_ZN_5trolls(uu) = sum((bb(:,uu)/trials).*P_Y'); % P(Remaining HP is N and 5 trolls slayed) located at index N of this vector
end

d = 1:2*Level;
Expected_HP_Remaining = sum((d).*(P_ZN_5trolls./P_5trolls_slayed)) % Expected HP of remaining troll
    