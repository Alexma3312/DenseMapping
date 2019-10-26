%%
clear;
image = zeros(100, 100, 3);
depth = zeros(100, 100);

for i = 1:20
    for j = 1:60
        image(i, j, :) = [100, 0, 0];
        depth(i, j) = 1;
    end
end

for i = 21:40
    for j = 1:60
        image(i, j, :) = [0, 100, 0];
        depth(i, j) = 10;
    end
end

for i = 41:60
    for j = 1:60
        image(i, j, :) = [0, 0, 100];
        depth(i, j) = 50;
    end
end

for i = 61:80
    for j = 1:60
        image(i, j, :) = [100, 100, 0];
        depth(i, j) = 100;
    end
end

for i = 81:100
    for j = 1:60
        image(i, j, :) = [0, 100, 100];
        depth(i, j) = 150;
    end
end

for i = 1:50
    for j = 61:100
        image(i, j, :) = [100, 0, 100];
        depth(i, j) = 200;
    end
end

for i = 50:100 % this is a feature, not a bug
    for j = 61:100
        image(i, j, :) = [100, 100, 100];
        depth(i, j) = 250;
    end
end

image = uint8(rgb2gray(image/256.0)*256);

figure(1);
imshow(image); 
imwrite(image,'test_image.png', 'png')

figure(2);
depth = uint8(depth);
imshow(depth);
imwrite(depth,'test_depth.png', 'png')

%%
clear;
image = uint8(zeros(27,5));
depth = image;

image(1:9+1,:) = 100;
image(10+1:18,:) = 0;
image(19:end,:) = 0;

depth(1:9,:) = 100;
depth(10:18-1,:) = 100;
depth(19-1:end,:) = 200;

figure(1);
imshow(image);
imwrite(image, 'test_image2.png', 'png');
figure(2);
imshow(depth);
imwrite(depth, 'test_depth2.png', 'png');