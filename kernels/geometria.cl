/*
    TRASLAZIONE 
    - x' = x + dx e y' = y + dy 
    - si applica l'inverse mapping, ovvero consideriamo i pixel di output x' per prendere il valore del pixel in input 
    - x = x' - dx e y = y' - dy
*/

__kernel void translation(__global const uchar* input, __global uchar* output, int rows, int cols, int dx, int dy)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    // inverse mapping, colcolo di x e y a partire da x' e y'
    int x_in = x-dx;
    int y_in = y-dy;

    
    // se pixel in input esce fuori dall'immagine, allora il pixel di output sarà nero
    if(x_in<0 || x_in>=cols || y_in<0 || y_in>=rows) 
    {
        output[y*cols+x] = 0;
        return;
    }

    output[y*cols+x] = input[y_in*cols+x_in];
}