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

for i = 50:100
    for j = 61:100
        image(i, j, :) = [100, 100, 100];
        depth(i, j) = 250;
    end
end

figure(1);
imshow(image); 
saveas(1, ['test_image.png'],'png');

figure(2);
imshow(uint8(depth));
saveas(2, ['test_depth.png'],'png');



