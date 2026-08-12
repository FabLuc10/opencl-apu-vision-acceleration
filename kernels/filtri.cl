// GAUSSIAN BLUR 
// si sfrutta la proprietà di separabilità della gaussiana per avere ottimizzazione computazionale
// complessità per pixel passa da O(N^2) a O(2N)

__constant float vettore_pesi[21] = {
    1.0f/1048576, 20.0f/1048576, 190.0f/1048576, 1140.0f/1048576, 
    4845.0f/1048576, 15504.0f/1048576, 38760.0f/1048576, 77520.0f/1048576, 
    125970.0f/1048576, 167960.0f/1048576, 184756.0f/1048576, 167960.0f/1048576, 
    125970.0f/1048576, 77520.0f/1048576, 38760.0f/1048576, 15504.0f/1048576, 
    4845.0f/1048576, 1140.0f/1048576, 190.0f/1048576, 20.0f/1048576, 1.0f/1048576
}; // pesi binomiali normalizzati che approssimano la funzione gaussiana 

__kernel void blur_x(__global const uchar* input, __global uchar* output, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    float somma = 0.0f;
    for(int k=-10; k<=10;k++)
    {
        int index = clamp(x+k,0,cols-1); // se coordinata x+k va sotto 0 viene forzata a 0, se va sopra cols-1 viene forzata a cols-1 
        somma += input[y*cols+index] * vettore_pesi[k+10];  // convoluzione spaziale 
    }

    output[y*cols+x] = (uchar) clamp(somma,0.0f,255.0f);
}

__kernel void blur_y(__global const uchar* input, __global uchar* output, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    float somma = 0.0f;
    for(int k=-10; k<=10;k++)
    {
        int index = clamp(y+k,0,rows-1);  
        somma += input[index*cols+x] * vettore_pesi[k+10];  
    }

    output[y*cols+x] = (uchar) clamp(somma,0.0f,255.0f);
}

// SOBEL per edge detection 
/*
    MASCHERA Gx per gradiente orizzontale   
    1  0 -1
    2  0 -2
    1  0 -1

    MASCHERA Gy per gradiente verticale   
    1  2  1
    0  0  0
   -1 -2 -1

*/
__kernel void sobel(__global const uchar* input, __global uchar* output, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x>=cols || y>=rows) return;

    // se pixel di bordo, si mette in output nero e si esce dal kernel 
    if (x == 0 || y == 0 || x == cols - 1 || y == rows - 1) {
        output[y * cols + x] = 0;
        return;
    }

    int inizio_riga_sopra = (y-1)*cols;
    int inizio_riga_mezzo = y*cols;
    int inizio_riga_sotto = (y+1)*cols;

    // lettura riga sopra
    int p00 = input[inizio_riga_sopra+x-1];
    int p01 = input[inizio_riga_sopra+x];
    int p02 = input[inizio_riga_sopra+x+1];

    // lettura riga in mezzo
    int p10 = input[inizio_riga_mezzo+x-1];
    int p12 = input[inizio_riga_mezzo+x+1];

    // lettura riga sotto
    int p20 = input[inizio_riga_sotto+x-1];
    int p21 = input[inizio_riga_sotto+x];
    int p22 = input[inizio_riga_sotto+x+1];

    int gx = (p00+2*p10+p20)-(p02+2*p12+p22);
    int gy = (p00+2*p01+p02)-(p20+2*p21+p22);

    // calcolo magnitudo approssimato 
    int magnitudo = clamp((int)(abs(gx)+abs(gy)),0,255);

    output[y*cols+x] = (uchar)magnitudo;
}