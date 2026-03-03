function y = soft(x,T)

T = T + eps;
y = max(abs(x) - T, 0);
y = y./(y+T) .* x;
% y = sign(x).*max(abs(x)-T,0);

