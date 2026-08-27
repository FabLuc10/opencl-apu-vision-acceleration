// GAUSSIAN BLUR 
// si sfrutta la proprietà di separabilità della gaussiana per avere ottimizzazione computazionale
// complessità per pixel passa da O(N^2) a O(2N)

// Memoria local formata nel seguente modo per blur_x ->         HALO         PIXEL(256)        HALO
//                                                          |xxxxxxxxxx|pixel dei work-items|xxxxxxxxxx|

#define R 10
__constant float vettore_pesi[21] = {
    1.0f/1048576, 20.0f/1048576, 190.0f/1048576, 1140.0f/1048576, 
    4845.0f/1048576, 15504.0f/1048576, 38760.0f/1048576, 77520.0f/1048576, 
    125970.0f/1048576, 167960.0f/1048576, 184756.0f/1048576, 167960.0f/1048576, 
    125970.0f/1048576, 77520.0f/1048576, 38760.0f/1048576, 15504.0f/1048576, 
    4845.0f/1048576, 1140.0f/1048576, 190.0f/1048576, 20.0f/1048576, 1.0f/1048576
}; // pesi binomiali normalizzati che approssimano la funzione gaussiana 


__kernel void blur_x(__global const uchar* input, __global uchar* output, int rows, int cols, __local uchar* locale)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int local_size = get_local_size(0);

    // work group ha dimensione 256x1, quindi se gy fuori significa che tutti i work-item di quel group sono fuori dall'immagine
    if(gy>=rows) return;

    // ogni work-item carica nelle memoria local il suo dato a paritre dalla posizione lx+R 
    locale[lx+R] = input[gy*cols+clamp(gx,0,cols-1)];

    // caricamento in memoria degli R vicini del work-group a sx e dx 
    if(lx<R)
        locale[lx] = input[gy*cols+clamp(gx-R,0,cols-1)]; // se coordinata x+k va sotto 0 viene forzata a 0, se va sopra cols-1 viene forzata a cols-1
    else if(lx>=local_size-R)
        locale[lx+2*R] = input[gy*cols+clamp(gx+R,0,cols-1)];

    // sincronizzazione: blocca l'avanzamento del work_item fino a quando tutti hanno finito di copiare nella memoria locale
    // garantisce consistenza dei dati
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gx>=cols) return;

    float somma = 0.0f;

    #pragma unroll
    for(int k=-R; k<=R;k++)
        somma += locale[lx+R+k] * vettore_pesi[k+R];  // convoluzione spaziale 
    
    output[gy*cols+gx] = (uchar) clamp(somma,0.0f,255.0f);
}

__kernel void blur_y(__global const uchar* input, __global uchar* output, int rows, int cols, __local uchar* locale)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int ly = get_local_id(1);
    int local_size = get_local_size(1);

    // work group ha dimensione 1x256, quindi se gx fuori significa che tutti i work-item di quel group sono fuori dall'immagine
    if(gx>=cols) return;

    locale[ly+R] = input[clamp(gy,0,rows-1)*cols+gx];

    if(ly<R)
        locale[ly] = input[clamp(gy-R,0,rows-1)*cols+gx]; 
    else if(ly>=local_size-R)
        locale[ly+2*R] = input[clamp(gy+R,0,rows-1)*cols+gx];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gy>=rows) return;

    float somma = 0.0f;

    #pragma unroll
    for(int k=-R; k<=R;k++)
        somma += locale[ly+R+k] * vettore_pesi[k+R];  
    
    output[gy*cols+gx] = (uchar) clamp(somma,0.0f,255.0f);
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

    // lettura riga di mezzo
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