clear
addpath('functions')

file_name = '../../data/ER_5_data.mat';
save_name = sprintf('%s_time_sdp', 'ER_correct_5_data');
load(file_name)

cw_len = 21;
nSample = 100;

n = size(coupling, 1);
A2 = zeros(n+1,n+1,cw_len,nSample);
time_sdp = zeros(cw_len, nSample);

for cw = 1:cw_len
for iter = 1:nSample
    A2(:,:,cw,iter) = [coupling(:,:,cw,iter),bias(:,cw,iter);bias(:,cw,iter)',0];
end    
end

Ux = zeros(n+1,n+1,cw_len,nSample);
Sx = zeros(n+1,n+1,cw_len,nSample);

for cw = 1:cw_len
for iter = 1:nSample
    fprintf('CW=%i, sample=%i\n ', cw, iter);
    tic
    [U,S,~] = mintraceNSD(A2(:,:,cw,iter));
    time_sdp(cw,iter) = toc;
    Ux(:,:,cw,iter) = U;
    Sx(:,:,cw,iter) = S;
end
end

save(save_name, 'time_sdp')
save('ER_correct_5_Ux_Sx','Ux','Sx')
