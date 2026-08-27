#include "AlgoritmiCPU.hpp"
#include <cmath>
#include <algorithm>

#define DIMENSIONE 7 
#define R ((DIMENSIONE - 1) / 2)
#define P 21

using namespace std;
using namespace cv;

const float vettore_pesi[P] = {
    1.0f/1048576, 20.0f/1048576, 190.0f/1048576, 1140.0f/1048576, 
    4845.0f/1048576, 15504.0f/1048576, 38760.0f/1048576, 77520.0f/1048576, 
    125970.0f/1048576, 167960.0f/1048576, 184756.0f/1048576, 167960.0f/1048576, 
    125970.0f/1048576, 77520.0f/1048576, 38760.0f/1048576, 15504.0f/1048576, 
    4845.0f/1048576, 1140.0f/1048576, 190.0f/1048576, 20.0f/1048576, 1.0f/1048576
};



static void controlloAllocazione(const Mat& input, Mat& output)
{
    if(output.empty() || input.size() != output.size() || input.type() != output.type()) {
        output = cv::Mat(input.size(), input.type());
    }
}


// FILTRI 

void runSobelCPU(const Mat& input, Mat& output)
{
    controlloAllocazione(input,output);

    // imposto tutti i pixel di bordo a 0 
    output.setTo(0);

    int rows = input.rows;
    int cols = input.cols;

    for(int y=1;y<rows-1;y++)
    {
        // puntatori alle righe di interesse in input 
        const uchar* riga_sopra = input.ptr<uchar>(y-1);
        const uchar* riga_corrente = input.ptr<uchar>(y);
        const uchar* riga_sotto = input.ptr<uchar>(y+1);

        // puntatore alla riga in output
        uchar* riga_output = output.ptr<uchar>(y);

        for(int x=1;x<cols-1;x++)
        {
            // lettura pixel della riga superiore
            int p00 = riga_sopra[x-1];
            int p01 = riga_sopra[x];
            int p02 = riga_sopra[x+1];

            // lettura pixel della riga di mezzo
            int p10 = riga_corrente[x-1];
            int p12 = riga_corrente[x+1];

            // lettura pixel riga inferiore
            int p20 = riga_sotto[x-1];
            int p21 = riga_sotto[x];
            int p22 = riga_sotto[x+1];

            int gx = (p00+2*p10+p20)-(p02+2*p12+p22);
            int gy = (p00+2*p01+p02)-(p20+2*p21+p22);

            // calcolo magnitudo 
            int magnitudo = clamp(abs(gx)+abs(gy),0,255);

            riga_output[x] = (uchar) magnitudo;
        }
    }

}

void runBlurCPU(const Mat& input, Mat& output)
{
    controlloAllocazione(input,output);
    
    int rows = input.rows;
    int cols = input.cols;
    Mat temp = Mat(input.size(), input.type());

    // Blur_x
    for(int y=0;y<rows;y++)
    {
        const uchar* riga_input = input.ptr<uchar>(y);
        uchar* riga_temp = temp.ptr<uchar>(y);

        // bordo sinistro
        for(int x=0;x<P/2;x++)
        {
            float somma = 0.0f;
            for(int k=-P/2;k<=P/2;k++) 
                somma += riga_input[clamp(x+k,0,cols-1)]*vettore_pesi[k+P/2];
            
            riga_temp[x] = (uchar) clamp(somma,0.0f,255.0f);
        }

        // area centrale 
        for(int x=P/2;x<cols-P/2;x++)
        {
            float somma = 0.0f;
            for(int k=-P/2; k<=P/2;k++) 
                somma += riga_input[x+k]*vettore_pesi[k+P/2];
            
            riga_temp[x] = (uchar) clamp(somma,0.0f,255.0f);
        }

        // bordo di destra
        for(int x=cols-P/2;x<cols;x++)
        {
            float somma = 0.0f;
            for(int k=-P/2;k<=P/2;k++) 
                somma += riga_input[clamp(x+k,0,cols-1)]*vettore_pesi[k+P/2];
            
            riga_temp[x] = (uchar) clamp(somma,0.0f,255.0f);
        }

    }

    // Blur_y
    
    // bordo superiore
    for (int y=0;y<P/2;y++) {
        uchar* riga_output = output.ptr<uchar>(y);

        for (int x=0;x<cols;x++) 
        {
            float somma = 0.0f;

            for (int k=-P/2;k<=P/2;k++) 
            {
                int index = clamp(y+k,0,rows-1);
                somma += temp.ptr<uchar>(index)[x]*vettore_pesi[k + P/2];
            }
            riga_output[x] = (uchar)clamp(somma,0.0f,255.0f);
        }
    }

    // area centrale
    for (int y=P/2;y<rows-P/2;y++) {
        uchar* riga_output = output.ptr<uchar>(y);

        for (int x=0;x<cols;x++) 
        {
            float somma = 0.0f;

            for (int k=-P/2;k<=P/2;k++) 
                somma += temp.ptr<uchar>(y+k)[x]*vettore_pesi[k+P/2];
            
            riga_output[x] = (uchar)clamp(somma,0.0f,255.0f);
        }
    }

    // bordo inferiore
    for (int y=rows-P/2;y<rows;y++) {

        uchar* riga_output = output.ptr<uchar>(y);
        for (int x=0; x<cols; x++) {
            float somma = 0.0f;

            for (int k =-P/2; k<=P/2; k++) {
                int index = clamp(y+k,0,rows-1);
                somma += temp.ptr<uchar>(index)[x] * vettore_pesi[k+P/2];
            }
            riga_output[x] = (uchar)clamp(somma,0.0f,255.0f);
        }
    }
}


// MORFOLOGIA

void runErosionCPU(const Mat& input, Mat& output)
{
    controlloAllocazione(input,output);
    int rows = input.rows;
    int cols = input.cols;

    
    for(int y=0;y<rows;y++)
    {
        uchar* riga_output = output.ptr<uchar>(y);

        for(int x=0;x<cols;x++)
        {
            uchar minimo = 255;
            for(int j=-R;j<=R;j++)
            {
                int y_clamp = clamp(y+j,0,rows-1);
                const uchar* riga_input = input.ptr<uchar>(y_clamp);

                for(int i=-R;i<=R;i++)
                {
                    int x_clamp = clamp(x+i,0,cols-1);
                    uchar intensita = riga_input[x_clamp];
                    minimo = min(minimo,intensita);
                }
            }
            riga_output[x]=minimo;
        }
    }
}

void runDilationCPU(const Mat& input, Mat& output)
{
    controlloAllocazione(input,output);
    int rows = input.rows;
    int cols = input.cols;

    
    for(int y=0;y<rows;y++)
    {
        uchar* riga_output = output.ptr<uchar>(y);

        for(int x=0;x<cols;x++)
        {
            uchar massimo = 0;
            for(int j=-R;j<=R;j++)
            {
                int y_clamp = clamp(y+j,0,rows-1);
                const uchar* riga_input = input.ptr<uchar>(y_clamp);

                for(int i=-R;i<=R;i++)
                {
                    int x_clamp = clamp(x+i,0,cols-1);
                    uchar intensita = riga_input[x_clamp];
                    massimo = max(massimo,intensita);
                }
            }
            riga_output[x]=massimo;
        }
    }
}

// GEOMETRIA 

static uchar interpolazione_bilineare(const uchar* input, float x, float y, int rows, int cols)
{
    if(x<0 || x>=cols || y<0 || y>=rows)
        return 0;
    
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = min(x0+1,cols-1);
    int y1 = min(y0+1,rows-1);
    
    float peso_x = x-x0;
    float peso_y = y-y0;

    int p0 = input[y0*cols+x0];
    int p1 = input[y0*cols+x1];
    int p2 = input[y1*cols+x0];
    int p3 = input[y1*cols+x1];

    float top = p0*(1-peso_x)+p1*peso_x;
    float bottom = p2*(1-peso_x)+p3*peso_x;

    return (uchar) (top*(1-peso_y)+bottom*peso_y);
    
}

void runTranslationCPU(const Mat& input, Mat& output, int dx, int dy)
{
    controlloAllocazione(input,output);
    int rows = input.rows;
    int cols = input.cols;

    const uchar* ptr_input = input.ptr<uchar>();

    for(int y=0;y<rows;y++)
    {
        uchar* riga_output = output.ptr<uchar>(y);
        for(int x=0;x<cols;x++)
        {
            int x_in = x-dx;
            int y_in = y-dy;

            if(x_in<0 || x_in>=cols || y_in<0 || y_in>=rows)
                riga_output[x] = 0;
            else
                riga_output[x] = ptr_input[y_in*cols+x_in];
        }       
    }
}

void runRotationCPU(const Mat& input, Mat& output, float grado_rotazione)
{
    controlloAllocazione(input,output);

    int rows = input.rows;
    int cols = input.cols;
    float rad = grado_rotazione*CV_PI/180.0f;
    float coseno = cos(-rad);
    float seno = sin(-rad);
    float cx = cols/2.0f;
    float cy = rows/2.0f;

    const uchar* ptr_input = input.ptr<uchar>();
    
    for(int y=0;y<rows;y++)
    {
        uchar* riga_output = output.ptr<uchar>(y);
        for(int x=0;x<cols;x++)
        {
            float dx = x-cx;
            float dy = y-cy;
            
            float x_in = (dx*coseno-dy*seno)+cx;
            float y_in = (dx*seno+dy*coseno)+cy;

            riga_output[x]=interpolazione_bilineare(ptr_input,x_in,y_in,rows,cols);
        }

    }
}



void runScalingCPU(const Mat& input, Mat& output, float scala)
{
    controlloAllocazione(input,output);
    int rows = input.rows;
    int cols = input.cols;

    float cx = cols/2.0f;
    float cy = rows/2.0f;

    const uchar* ptr_input = input.ptr<uchar>();

    for(int y=0;y<rows;y++)
    {  
        uchar* riga_output = output.ptr<uchar>(y);
        for(int x=0;x<cols;x++)
        {
            float x_in = (x-cx)/scala + cx;
            float y_in = (y-cy)/scala + cy;
            riga_output[x]=interpolazione_bilineare(ptr_input,x_in,y_in,rows,cols);
        }
    }
}

