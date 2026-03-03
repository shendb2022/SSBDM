function super_pixel = find_pixels(X, num, rows, cols, nbands)
super_pixel = zeros(num, nbands);
for j = 1:num
  super_pixel(j,:) = X(rows(j),cols(j),:);
end