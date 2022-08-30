function cor_ADMM(n,C,HL,HU,Tol)

Max_It=500;
beta=10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% initial values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X00=eye(n,n); Y00=eye(n,n); Z00=zeros(n,n);
X0=(Y00+Z00+C)/2; [V,D]=eig(X0);D=max(0,D); X0=(V*D)*V';
EX=X0-X00; ex=max(max(abs(EX)));
Y0=min(max((X0-Z00+C)/2,HL),HU); EY=Y0-Y00; ey=max(max(abs(EY)));
Z0=Z00-(X0-Y0); EZ=Z0-Z00; ez=max(max(abs(EZ)));
Txyz0=max(max(ex,ey),ez);

for i=1:Max_It
    if ez<=Tol*0.9
        beta=max(beta*2/3,0.6);
    end
    X=(Y0*beta+Z0+C)/(1+beta);
    [V,D]=eig(X); D=max(0,D);
    X=(V*D)*V'; 
    
    Y=min(max((X*beta-Z0+C)/(1+beta),HL),HU);
    ey=max(max(abs(Y-Y0)));
    
    Z=Z0-(X-Y)*beta;
    ez=max(max(abs(X-Y)));
    Tyz=max(ey,ez);
    err=Tyz/Txyz0
    
    if err<Tol
        disp('Find the solution')
        err
        i
        break;
    else
        Y0=Y;Z0=Z;
    end
end
