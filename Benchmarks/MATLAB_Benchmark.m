% Benchmarking MATLAB Performance
disp('Testing some linear algebra functions');
i=1000;
disp('Eig');tic;data=rand(i,i);eig(data);toc;
disp('Svd');tic;data=rand(i,i);[u,s,v]=svd(data);s=svd(data);toc;
disp('Inv');tic;data=rand(i,i);result=inv(data);toc;
disp('Det');tic;data=rand(i,i);result=det(data);toc;
disp('Dot');tic;a=rand(i,i);b=inv(a);result=a*b-eye(i);toc;
disp('Done');