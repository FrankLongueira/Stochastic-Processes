% Frank Longueira & Matthew Smarsch
% Stochastic Processes - MATLAB Project 2

clear all;
close all;
clc;

%% Problem 1: Radar Detection
Var_X = 3;
A = 8;
P_Target_Present = 0.2;
P_Target_Absent = 0.8;
Sigma = sqrt(Var_X);

P_Y_Given_Present = @(y) (1/sqrt(Var_X*2*pi))*exp(-((y-A)^2)/(2*Var_X));

P_Y_Given_Absent = @(y) (1/sqrt(Var_X*2*pi))*exp(-((y)^2)/(2*Var_X));

bucket = 0; % Bucket to catch the errors
trials = 1e5;

% Part (a)

% Simulated Probability of Error for Different Mean, Same Variances

for i = 1:trials
    choose = randi(5,1,1);
    if choose == 1 % Target is present
        y = A + normrnd(0,sqrt(Var_X));
        if P_Y_Given_Present(y)*P_Target_Present >= P_Y_Given_Absent(y)*P_Target_Absent
            bucket = bucket; % Guess is correct, do nothing
        else
            bucket = bucket + 1; % Guess is wrong, increment bucket
        end
    else           % Target is absent
        y = normrnd(0,sqrt(Var_X));
        if P_Y_Given_Present(y)*P_Target_Present >= P_Y_Given_Absent(y)*P_Target_Absent
            bucket = bucket + 1; % Guess is wrong, increment bucket
        else
            bucket = bucket; % Guess is correct, do nothing
        end
    end
end

Simulated_Probability_Error = bucket/trials % Approximates the theoretical error very accurately

% Theoretical Probability of Error for Different Mean, Same Variances


x = (2*Var_X*log(4)+A^2)/(2*A);
P_Choose_Absent_Given_Present = normcdf(x,A,Sigma);
P_Choose_Present_Given_Absent = 1 -  normcdf(x,0,Sigma);

Theoretical_Probability_Error = (P_Choose_Absent_Given_Present)*(P_Target_Present) + (P_Choose_Present_Given_Absent)*(P_Target_Absent)

%% Part (b)
eta1 = 0:0.001:1000;
L = length(eta1);
P_D_Total = zeros(5,L);
P_F_Total = zeros(5,L);
n = 0;

% Theoretical ROC's
for A = 1:2:5
    n = n + 1;
    x = (2*Var_X*log(eta1)+A^2)/(2*A);
    P_D_Total(n,:) = 1-normcdf(x,A,Sigma);
    P_F_Total(n,:) = 1-normcdf(x,0,Sigma);
end


%% Simulated ROC's 

trials = 1e3;
k = 0;
eta2 = 0:0.4:500;
L = length(eta2);
Num_SNRs = 3;
PD_Per_SNR = zeros(Num_SNRs,L);
PF_Per_SNR = zeros(Num_SNRs,L);

for A = 1:2:5
    P_Y_Given_Present = @(y) (1/sqrt(Var_X*2*pi))*exp(-((y-A)^2)/(2*Var_X));
    P_Y_Given_Absent = @(y) (1/sqrt(Var_X*2*pi))*exp(-((y)^2)/(2*Var_X));
    PD_Per_Eta = zeros(1,L);
    PF_Per_Eta = zeros(1,L);
    k = k+1;
    n = 0;
    for eta = eta2
        bucket_H1_H1 = 0;
        bucket_H1_HO = 0;
        n = n+1;
        choose_pres = 0;
        choose_abs = 0;
        for i = 1:trials
            choose = randi(5,1,1);
            
            if choose == 1 % Target is present
                choose_pres = choose_pres + 1;
                y = A + normrnd(0,sqrt(Var_X));
                if P_Y_Given_Present(y) >= P_Y_Given_Absent(y)*eta
                    bucket_H1_H1 = bucket_H1_H1 + 1; % Detections given present
                end
            else           % Target is absent
                choose_abs = choose_abs + 1;
                y = normrnd(0,sqrt(Var_X));
                if P_Y_Given_Present(y) >= P_Y_Given_Absent(y)*eta
                    bucket_H1_HO = bucket_H1_HO + 1; % False alarm given absent
                end
            end
        end
        PD = bucket_H1_H1/choose_pres;
        PF = bucket_H1_HO/choose_abs;
        PD_Per_Eta(1,n)= PD;
        PF_Per_Eta(1,n)= PF;
    end
    PD_Per_SNR(k,:)= PD_Per_Eta;
    PF_Per_SNR(k,:)= PF_Per_Eta;
end

% Plotting Theoretical and Simulated ROCs

figure;
subplot(2,1,1)
plot(P_F_Total(1,:), P_D_Total(1,:),'b',P_F_Total(2,:), P_D_Total(2,:),'r',P_F_Total(3,:), P_D_Total(3,:),'g')
    

xlabel('Probability of False Alarm')
ylabel('Probability of Detection')
title('Theoretical ROC for Various SNRs')
legend('SNR = 0.33', 'SNR = 1', 'SNR = 1.67','Location','southeast')

subplot(2,1,2)
plot(smooth(PF_Per_SNR(1,:)),smooth(PD_Per_SNR(1,:)),'b',smooth(PF_Per_SNR(2,:)),smooth(PD_Per_SNR(2,:)),'r',...
    smooth(PF_Per_SNR(3,:)),smooth(PD_Per_SNR(3,:)),'g')

xlabel('Probability of False Alarm')
ylabel('Probability of Detection')
title('Simulated ROCs for Various SNRs')
legend('SNR = 0.33', 'SNR = 1', 'SNR = 1.67','Location','southeast')

%% Part (c)

% Decision Rule is P_Y_Given_Present(y)*P_Target_Present (H1)>< (H0)P_Y_Given_Absent(y)*P_Target_Absent/10

A = 3; % SNR is 1

loc1 = find(eta1 == 4/10);
loc2 = find(eta2 == 4/10);

% Theoretical ROC
figure;
subplot(2,1,1)
plot(P_F_Total(2,:), P_D_Total(2,:),'r',P_F_Total(2,:), ones(length(eta1),1)*P_D_Total(2,loc1),'b');

xlabel('Probability of False Alarm')
ylabel('Probability of Detection')
title('Location of \eta = 0.4 for Theoretical ROC at SNR = 1')
legend('SNR = 1', 'Location of \eta = 0.4','Location','southeast')

% Simulated ROC

subplot(2,1,2)
plot(smooth(PF_Per_SNR(2,:)),smooth(PD_Per_SNR(2,:)),'r',smooth(PF_Per_SNR(2,:)), smooth(ones(length(eta2),1)*PD_Per_SNR(2,loc2)),'b');

xlabel('Probability of False Alarm')
ylabel('Probability of Detection')
title('Location of \eta = 0.4 for Simulated ROC at SNR = 1')
legend('SNR = 1', 'Location of \eta = 0.4','Location','southeast')




%% Part (d)

A = 3; % SNR is 1
C01 = 10;
C10 = 1;
Expected_Cost = zeros(1,101);
n = 0;
% Theoretical Expected Cost

P_Target_Present = 0:0.01:1;
    
   
P_Target_Absent = 1-P_Target_Present;
eta = P_Target_Absent./((C01)*P_Target_Present);
x = (2*Var_X*log(eta)+A^2)/(2*A);
P_D = 1-normcdf(x,A,Sigma);
P_F = 1-normcdf(x,0,Sigma);
Expected_Cost = C01*P_Target_Present + C10*P_Target_Absent.*P_F - (C01)*P_Target_Present.*P_D;
    

%% Part (e)
% Calculating eta_minimax and the associated expected cost

PD = 1 - (P_F_Total(2,:)/10);

Location_eta_minimax = find((abs(PD - P_D_Total(2,:)))<=0.0002);
eta_minimax = eta1(Location_eta_minimax);
P1_minimax = 1/(10*eta_minimax+1);

Expected_Cost_minimax = Expected_Cost(find(P_Target_Present == round(P1_minimax*100)/100));

% Expected_Cost_minimax  plotted
figure;
plot(P_F_Total(2,:), P_D_Total(2,:),'g',smooth(PF_Per_SNR(2,:)),smooth(PD_Per_SNR(2,:)),'r',P_F_Total(2,:),PD,'b')

xlabel('Probability of False Alarm')
ylabel('Probability of Detection')
title('Location of \etaminimax = 0.29 for ROC at SNR = 1')
legend('Theoretical ROC at SNR = 1','Simulated ROC at SNR = 1', 'Location of \eta-minimax ','Location','southeast')


figure;
plot(P_Target_Present, Expected_Cost, 'g',P1_minimax,Expected_Cost_minimax,'r*' )

xlabel('Probability of Target Present')
ylabel('Expected Cost')
title('Expected Cost Over Different Priors at SNR = 1 with Minimax Cost Plotted')
legend('Optimal Expected Cost','Minimax Expected Cost','Location','southeast')


%% Part (f)

Var_X = 1;
Var_Z = 25;
A = 1;
P_Target_Present = 0.2;
P_Target_Absent = 0.8;
Sigma_X = sqrt(Var_X);
Sigma_Z = sqrt(Var_Z);
eta = P_Target_Absent/P_Target_Present;

P_Y_Given_Present = @(y) (1/sqrt(Var_X*2*pi))*exp(-((y-A)^2)/(2*Var_X));
P_Y_Given_Absent = @(y) (1/sqrt(Var_Z*2*pi))*exp(-((y-A)^2)/(2*Var_Z));

bucket = 0; % Bucket to catch the errors
trials = 1e5;

    % Part (a)
    % Simulated Probability of Error for Same Mean, Different Variances
for i = 1:trials
    choose = randi(5,1,1);
    if choose == 1 % Target is present
        y = A + normrnd(0,sqrt(Var_X));
        if P_Y_Given_Present(y)*P_Target_Present >= P_Y_Given_Absent(y)*P_Target_Absent
            bucket = bucket; % Guess is correct, do nothing
        else
            bucket = bucket + 1; % Guess is wrong, increment bucket
        end
    else           % Target is absent
        y = A + normrnd(0,sqrt(Var_Z));
        if P_Y_Given_Present(y)*P_Target_Present >= P_Y_Given_Absent(y)*P_Target_Absent
            bucket = bucket + 1; % Guess is wrong, increment bucket
        else
            bucket = bucket; % Guess is correct, do nothing
        end
    end
end

Simulated_Probability_Error = bucket/trials % Approximates the theoretical error very accurately

% Theoretical Probability of Error for Same Mean, Different Variances

if (Sigma_Z/Sigma_X) <= eta
    P_Choose_Absent_Given_Present = 1;
    P_Choose_Present_Given_Absent = 0;

else
    x = sqrt(((2*Var_Z*Var_X/(Var_X-Var_Z))*log((Sigma_X/Sigma_Z)*eta)));
    P_Choose_Absent_Given_Present = 2*(1-normcdf(x,0,Sigma_X));
    P_Choose_Present_Given_Absent = normcdf(x,0,Sigma_Z)-normcdf(-x,0,Sigma_Z);

end
    Theoretical_Probability_Error = (P_Choose_Absent_Given_Present)*(P_Target_Present) + (P_Choose_Present_Given_Absent)*(P_Target_Absent)
    
    %% Part (b)
    
    % Theoretical ROC Calculation
k=0;
Num_Ratios = 3;
eta_vect = 1e-50:0.001:1000;
Num_eta = length(eta_vect);
P_D_Eta = zeros(1,Num_eta);
P_F_Eta = zeros(1,Num_eta);
P_D_Var = zeros(Num_Ratios,Num_eta);
P_F_Var = zeros(Num_Ratios,Num_eta);

for Var_Z = 1.5:1:3.5
    k=k+1;
    n=0;
    for eta = eta_vect
        n=n+1;
        
        if (sqrt(Var_Z)/Sigma_X) <= eta
            P_D = 0;
            P_F = 0;

        else
            x = sqrt(((2*Var_Z*Var_X/(Var_X-Var_Z))*log((Sigma_X/Sigma_Z)*eta)));
            P_D = normcdf(x,0,Sigma_X)-normcdf(-x,0,Sigma_X);
            P_F = normcdf(x,0,Sigma_Z)-normcdf(-x,0,Sigma_Z);
        end
        P_D_Eta(1,n) = P_D;
        P_F_Eta(1,n) = P_F;
    end
    P_D_Var(k,:) = P_D_Eta(1,:);
    P_F_Var(k,:) = P_F_Eta(1,:);      
end


    %% Simulated ROCs


trials = 1e3;
k = 0;
eta2 = 0:0.5:1000;
L = length(eta2);
Num_Vars = 3;
PD_Per_SNR = zeros(Num_Vars,L);
PF_Per_SNR = zeros(Num_Vars,L);

for Var_Z = 1.5:1:3.5
    P_Y_Given_Present = @(y) (1/sqrt(Var_X*2*pi))*exp(-((y-A)^2)/(2*Var_X));
    P_Y_Given_Absent = @(y) (1/sqrt(Var_Z*2*pi))*exp(-((y-A)^2)/(2*Var_Z));
    PD_Per_Eta = zeros(1,L);
    PF_Per_Eta = zeros(1,L);
    k = k+1;
    n = 0;
    for eta = eta2
        bucket_H1_H1 = 0;
        bucket_H1_HO = 0;
        n = n+1;
        choose_pres = 0;
        choose_abs = 0;
        for i = 1:trials
            choose = randi(5,1,1);
            
            if choose == 1 % Target is present
                choose_pres = choose_pres + 1;
                y = A + normrnd(0,sqrt(Var_X));
                if P_Y_Given_Present(y) >= P_Y_Given_Absent(y)*eta
                    bucket_H1_H1 = bucket_H1_H1 + 1; % Detections given present
                end
            else           % Target is absent
                choose_abs = choose_abs + 1;
                y = A + normrnd(0,sqrt(Var_Z));
                if P_Y_Given_Present(y) >= P_Y_Given_Absent(y)*eta
                    bucket_H1_HO = bucket_H1_HO + 1; % False alarm given absent
                end
            end
        end
        PD = bucket_H1_H1/choose_pres;
        PF = bucket_H1_HO/choose_abs;
        PD_Per_Eta(1,n)= PD;
        PF_Per_Eta(1,n)= PF;
    end
    PD_Per_SNR(k,:)= PD_Per_Eta;
    PF_Per_SNR(k,:)= PF_Per_Eta;
end


figure
subplot(2,1,1)
plot(P_F_Var(1,:),P_D_Var(1,:),'b',P_F_Var(2,:),P_D_Var(2,:),'g',...
    P_F_Var(3,:),P_D_Var(3,:),'r')
xlabel('Probability of False Alarm')
ylabel('Probability of Detection')
title('Theoretical ROC for Various Variance Ratios')
legend('Variance Ratio = 1.5', 'Variance Ratio = 2.5', 'Variance Ratio = 3.5','Location','southeast')

subplot(2,1,2)
plot(smooth(PF_Per_SNR(1,:)),smooth(PD_Per_SNR(1,:)),'b',smooth(PF_Per_SNR(2,:)),smooth(PD_Per_SNR(2,:)),'g',...
    smooth(PF_Per_SNR(3,:)),smooth(PD_Per_SNR(3,:)),'r')

xlabel('Probability of False Alarm')
ylabel('Probability of Detection')
title('Simulated ROCs for Various Variance Ratios')
legend('Variance Ratio = 1.5', 'Variance Ratio = 2.5', 'Variance Ratio = 3.5','Location','southeast')

%% Problem 3: Classifier

clear all;
clc;

load('Iris.mat'); % Load data into MATLAB

length(find(labels==1)); % Each class has 1/3 probability of occurring
length(find(labels==2));
length(find(labels==3));

num_trials = 1e2;
Probability_Error_Each_Trial = zeros(1,num_trials);
Confusion = zeros(3,3);
for trials = 1:num_trials
    % Splitting data into classes
    features_1 = features(1:50,:); 
    features_2 = features(51:100,:);
    features_3 = features(101:150,:);


    % Random splitting in half for each class
    Split_1 = randperm(50);
    Split_1 = Split_1(1:25);
    Split_2 = randperm(50);
    Split_2 = Split_2(1:25);
    Split_3 = randperm(50);
    Split_3 = Split_3(1:25);


    % Training Data Chosen
    Class_1_Training = features_1(Split_1,:);
    Class_2_Training = features_2(Split_2,:);
    Class_3_Training = features_3(Split_3,:);

    % Class 1 Mean and Covariance Estimation
    mu1 = mean(Class_1_Training);   % Each column is the estimated mean of the specific feature
    cov1 = cov(Class_1_Training,1); % Estimated covariance matrix

    % Class 2 Mean and Covariance Estimation
    mu2 = mean(Class_2_Training);   % Each column is the estimated mean of the specific feature
    cov2 = cov(Class_2_Training,1); % Estimated covariance matrix


    % Class 3 Mean and Covariance Estimation
    mu3 = mean(Class_3_Training);   % Each column is the estimated mean of the specific feature
    cov3 = cov(Class_3_Training,1); % Estimated covariance matrix


    % Testing Data
    features_1(Split_1,:) = [];
    features_2(Split_2,:) = [];
    features_3(Split_3,:) = [];

    Test_Data = [features_1;features_2;features_3];
    order = randperm(75);
    Test_Data = Test_Data(order,:); % Shuffle data

    [m p] = size(Test_Data);
    Errors = 0;
    
    for n = 1:m
        Test = Test_Data(n,:);        
        if (mvnpdf(Test,mu1,cov1)/mvnpdf(Test,mu2,cov2) >= 1) && (mvnpdf(Test,mu1,cov1)/mvnpdf(Test,mu3,cov3) >= 1)
            Class_1_Classify(n,:) = Test;
            if sum(ismember(features(1:50,:),Test,'rows')) >= 1
                Confusion(1,1) = Confusion(1,1)+1; % Predicted Class 1 and correct 
            elseif sum(ismember(features(51:100,:),Test,'rows')) >= 1
                Confusion(2,1) = Confusion(2,1)+1; % Predicted Class 1 but Class 2
                Errors = Errors + 1;
            else
                Confusion(3,1) = Confusion(3,1)+1; % Predicted Class 1 but Class 3
                Errors = Errors + 1;
            end
    
        elseif (mvnpdf(Test,mu2,cov2)/mvnpdf(Test,mu1,cov1) >= 1) && (mvnpdf(Test,mu2,cov2)/mvnpdf(Test,mu3,cov3) >= 1)
                Class_2_Classify(n,:) = Test;
            if sum(ismember(features(1:50,:),Test,'rows')) >= 1
                Confusion(1,2) = Confusion(1,2)+1; % Predicted Class 2 but Class 1
                Errors = Errors + 1;
            elseif sum(ismember(features(51:100,:),Test,'rows')) >=1
                Confusion(2,2) = Confusion(2,2)+1; % Predicted Class 2 and correct
            else
                Confusion(3,2) = Confusion(3,2)+1; % Predicted Class 2 but Class 3
                Errors = Errors + 1;
            end
        else
            Class_3_Classify(n,:) = Test;
             if sum(ismember(features(1:50,:),Test,'rows')) >= 1
                Confusion(1,3) = Confusion(1,3)+1; % Predicted Class 3 but Class 1
                Errors = Errors + 1;
            elseif sum(ismember(features(51:100,:),Test,'rows')) >= 1
                Confusion(2,3) = Confusion(2,3)+1; % Predicted Class 3 but Class 2
                Errors = Errors + 1;
            else
                Confusion(3,3) = Confusion(3,3)+1; % Predicted Class 3 and correct
            end
        end
    end
    Probability_Error_Each_Trial(1,trials) = Errors/m;
end
Class_1_Classify(find(ismember(Class_1_Classify, zeros(1,4),'rows')==1),:)= []; % Getting rid of zero rows for presentation
Class_2_Classify(find(ismember(Class_2_Classify, zeros(1,4),'rows')==1),:)= [];
Class_3_Classify(find(ismember(Class_3_Classify, zeros(1,4),'rows')==1),:)= [];
 
Class_1_Classify; % Data classifed as Class 1
Class_2_Classify; % Data classifed as Class 2
Class_3_Classify; % Data classifed as Class 3

Confusion

Total_Probability_of_Error= mean(Probability_Error_Each_Trial)