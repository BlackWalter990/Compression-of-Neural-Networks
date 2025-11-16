%% This code implements the actual DA algorithm where we increase the 
% number of clusters when the condition for phase transition is satisfied.

%% Constructing Data Set 6
C1 = [0 0]; C2 = [1 0]; C3 = [0.5 0.9]; 
C4 = [5 0]; C5 = [6 0]; C6 = [5.5 0.9]; 
C7 = [2.5 4.2]; C8 = [3.5 4.2]; C9 = [3 5];
rng('default');
Centers = [C1; C2; C3; C4; C5; C6; C7; C8; C9];
Np = 200;
count = 1;
X = zeros(size(Centers,1)*Np, 2);
for i = 1 : size(Centers,1)
    for j = 1 : Np
        x = normrnd(Centers(i,1),0.125);
        y = normrnd(Centers(i,2),0.125);
        X(count,:) = [x y];
        count = count + 1;
    end
end
scatter(X(:,1),X(:,2),'.');
[M, N] = size(X);

%% Setting for DA parameters

K = 9; Tmin = 0.05; alpha = 0.99; PERTURB = 0.005; STOP = 1e-5;
T = 80; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);

while T >= Tmin
    L_old = inf;
    while 1
        [D,D_Act] = distortion(X,Y,M,N,K);
        num = exp(-D/T);
        den = repmat(sum(num,2),[1 K]);
        P = num./den;
        Py = P'*Px;
        Y = P'*(X.*repmat(Px,[1 N]))./repmat(Py,[1 N]) + PERTURB*rand(size(Y));
        L = -T*Px'*log(sum(exp(-D/T),2));
        Lu = -T*Px'*log(sum(exp(-D_Act/T),2));
        if(norm(L-L_old) < STOP)
            break;
        end
        L_old = L;
    end
    T = T*alpha;
end
hold on;
scatter(Y(:,1),Y(:,2),'d','filled'); hold off;
