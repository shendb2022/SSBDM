function X = inject_pixels(X, super_pixel, num, rows, cols)
for j = 1:num
   X(rows(j),cols(j),:) = super_pixel(j,:);
end