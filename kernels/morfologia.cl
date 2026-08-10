// EROSIONE 
// elemento strutturante piatto 3x3 
// 1 1 1 
// 1 1 1 
// 1 1 1
__kernel void erosion(__global const uchar* input, __global uchar* output, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    uchar minimo = 255;
    for(int i=-1;i<=1;i++)
        for(int j=-1;j<=1;j++)
        {
            int x_clamp = clamp(x+i,0,cols-1); 
            int y_clamp = clamp(y+j,0,rows-1); 

            uchar intensita = input[y_clamp*cols+x_clamp];
            minimo = min(minimo,intensita);
        }
    
    output[y*cols+x] = minimo;
}



// DILATAZIONE
// elemento strutturante piatto 3x3 
// 1 1 1 
// 1 1 1 
// 1 1 1

__kernel void dilation(__global const uchar* input, __global uchar* output, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    uchar massimo = 0;

    for(int i=-1;i<=1;i++)
        for(int j=-1;j<=1;j++)
        {
            int x_clamp = clamp(x+i,0,cols-1);
            int y_clamp = clamp(y+j,0,rows-1);

            uchar intensita = input[y_clamp*cols+x_clamp];
            massimo = max(massimo,intensita);
        }
    output[y*cols+x] = massimo;
}