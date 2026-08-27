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

// ROTAZIONE 
// x' = x cos(a) - y sen(a)
// y' = x sen(a) + y sen(a)

uchar interpolazione_bilineare(__global const uchar*input, float x, float y, int rows, int cols)
{
    if(x<0 || y<0 || x>cols-1 || y>rows-1)
        return 0;

    // individuo i 4 pixel vicini e faccio media pesata dell'intensità in base alla distanza
    int x0 = (int) x;
    int y0 = (int) y;
    int x1 = min(x0+1,cols-1);
    int y1 = min(y0+1,rows-1);

    // peso dato dalla distanza del punto dal punto arrotondato con floor 
    float peso_x = x-x0;
    float peso_y = y-y0;

    // recupero intensità pixel dei vicini 
    int p0 = input[y0*cols+x0]; // alto sx
    int p1 = input[y0*cols+x1]; // alto dx
    int p2 = input[y1*cols+x0]; // basso sx
    int p3 = input[y1*cols+x1]; // basso dx

    // interpolazione 
    float top = p0*(1-peso_x)+p1*peso_x; // media pesata tra i pixel adiacenti della riga superiore, più sono vicino a p0 più peso ha quel pixel
    float bottom = p2*(1-peso_x)+p3*peso_x; // media pesata tra i pixel adiacenti della riga sottostante
 
    return (uchar) (top*(1-peso_y)+bottom*peso_y); // media pesata tra i valori precendenti in base a se pixel originale più vicino alla riga di sopra o sotto

}

__kernel void rotation(__global const uchar* input, __global uchar* output, int rows, int cols, float grado_rotazione)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    float rad = grado_rotazione*M_PI_F/180.0f;

    //coordinate del centro 
    float cx = cols/2.0f;
    float cy = rows/2.0f;

    // inverse mapping 
    float dx = x-cx;
    float dy = y-cy;
    float coseno = cos(-rad);
    float seno = sin(-rad);

    // faccio rotazione di -alpha rispetto al centro dell'immagine e poi traslo per avere coordinate rispetto a (0,0) dell'immagine
    float x_in = (dx*coseno-dy*seno)+cx;
    float y_in = (dx*seno+dy*coseno)+cy;

    output[y*cols+x] = interpolazione_bilineare(input,x_in,y_in,rows,cols);
}

// SCALING 
// x' = x*scala
// y' = y*scala

__kernel void scaling(__global const uchar* input, __global uchar* output, int rows, int cols, float scala)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    float cx = cols/2.0f;
    float cy = rows/2.0f;

    float x_in = (x-cx)/scala+cx;
    float y_in = (y-cy)/scala+cy;

    output[y*cols+x] = interpolazione_bilineare(input, x_in, y_in, rows, cols);

}