#define DIMENSIONE 7 
#define R ((DIMENSIONE - 1) / 2)

// EROSIONE 
// elemento strutturante piatto RxR formato da tutti 1 
__kernel void erosion(__global const uchar* input, __global uchar* output, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    
    uchar minimo = 255;
    #pragma unroll 
    for(int j=-R;j<=R;j++)
    {
        int y_clamp = clamp(y+j,0,rows-1); 

        #pragma unroll 
        for(int i=-R;i<=R;i++)
        {
            int x_clamp = clamp(x+i,0,cols-1); 
            
            uchar intensita = input[y_clamp*cols+x_clamp];
            minimo = min(minimo,intensita);
        }
    }
    
    output[y*cols+x] = minimo;
}

// DILATAZIONE
// elemento strutturante piatto RxR formato da tutti 1

__kernel void dilation(__global const uchar* input, __global uchar* output, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    uchar massimo = 0;

    #pragma unroll
    for(int j=-R;j<=R;j++)
    {
        int y_clamp = clamp(y+j,0,rows-1);

        #pragma unroll
        for(int i=-R;i<=R;i++)
        {
            int x_clamp = clamp(x+i,0,cols-1);
            uchar intensita = input[y_clamp*cols+x_clamp];
            massimo = max(massimo,intensita);
        }
    }
    output[y*cols+x] = massimo;
}