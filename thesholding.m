function OUT = thesholding(INP, th)
[m,n] = size(INP);
OUT = zeros(m,n);
for i=1:m
    for j=1:n
        if(INP(i,j)<th)
            OUT(i,j) = 0;
        else
            OUT(i,j) = 255;
        end
    end
end
OUT = uint8(OUT);
       