#include "AlgoritmiCPU.hpp"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

const float vettore_pesi[21] = {
    1.0f/1048576, 20.0f/1048576, 190.0f/1048576, 1140.0f/1048576, 
    4845.0f/1048576, 15504.0f/1048576, 38760.0f/1048576, 77520.0f/1048576, 
    125970.0f/1048576, 167960.0f/1048576, 184756.0f/1048576, 167960.0f/1048576, 
    125970.0f/1048576, 77520.0f/1048576, 38760.0f/1048576, 15504.0f/1048576, 
    4845.0f/1048576, 1140.0f/1048576, 190.0f/1048576, 20.0f/1048576, 1.0f/1048576
};

const int raggio = 15;

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

    int rows = input.rows;
    int cols = input.cols;

    for(int y=0;y<rows;y++)
    {
        for(int x=0;x<cols;x++)
        {
            // controllo pixel di bordo
            if(x==0 || x==cols-1 || y==0 || y==rows-1)
            {
                output.at<uchar>(y,x)=0;
                continue;
            }
            
            // lettura pixel della riga superiore
            int p00 = input.at<uchar>(y-1,x-1);
            int p01 = input.at<uchar>(y-1,x);
            int p02 = input.at<uchar>(y-1,x+1);

            // lettura pixel della riga di mezzo
            int p10 = input.at<uchar>(y,x-1);
            int p12 =input.at<uchar>(y,x+1);

            // lettura pixel riga inferiore
            int p20 = input.at<uchar>(y+1,x-1);
            int p21 = input.at<uchar>(y+1,x);
            int p22 = input.at<uchar>(y+1,x+1);

            int gx = (p00+2*p10+p20)-(p02+2*p12+p22);
            int gy = (p00+2*p01+p02)-(p20+2*p21+p22);

            // calcolo magnitudo 
            int magnitudo = clamp(abs(gx)+abs(gy),0,255);
            output.at<uchar>(y,x) = (uchar) magnitudo;
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
        for(int x=0;x<cols;x++)
        {
            float somma = 0.0f;
            for(int k=-10;k<=10;k++)
            {
                int index = clamp(x+k,0,cols-1);
                somma += input.at<uchar>(y,index)*vettore_pesi[k+10];
            }

            temp.at<uchar>(y,x) = (uchar) clamp(somma,0.0f,255.0f);
        }
    }

    // Blur_y
    for(int y=0;y<rows;y++)
    {
        for(int x=0;x<cols;x++)
        {
            float somma = 0.0f;
            for(int k=-10;k<=10;k++)
            {
                int index = clamp(y+k,0,rows-1);
                somma += temp.at<uchar>(index,x)*vettore_pesi[k+10];
            }

            output.at<uchar>(y,x) = (uchar) clamp(somma,0.0f,255.0f);
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
        for(int x=0;x<cols;x++)
        {
            uchar minimo = 255;
            for(int i=-raggio;i<=raggio;i++)
            {
                for(int j=-raggio;j<=raggio;j++)
                {
                    int x_clamp = clamp(x+i,0,cols-1);
                    int y_clamp = clamp(y+j,0,rows-1);
                    uchar intensita = input.at<uchar>(y_clamp,x_clamp);
                    minimo = min(minimo,intensita);
                }
            }
            output.at<uchar>(y,x)=minimo;
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
        for(int x=0;x<cols;x++)
        {
            uchar massimo = 0;
            for(int i=-raggio;i<=raggio;i++)
            {
                for(int j=-raggio;j<=raggio;j++)
                {
                    int x_clamp = clamp(x+i,0,cols-1);
                    int y_clamp = clamp(y+j,0,rows-1);
                    uchar intensita = input.at<uchar>(y_clamp,x_clamp);
                    massimo = max(massimo,intensita);
                }
            }
            output.at<uchar>(y,x)=massimo;
        }
    }
}

// GEOMETRIA 

static uchar interpolazione_bilineare(const Mat& input, float x, float y, int rows, int cols)
{
    if(x<0 || x>=cols || y<0 || y>=rows)
        return 0;
    
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = min(x0+1,cols-1);
    int y1 = min(y0+1,rows-1);
    
    float peso_x = x-x0;
    float peso_y = y-y0;

    int p0 = input.at<uchar>(y0,x0);
    int p1 = input.at<uchar>(y0,x1);
    int p2 = input.at<uchar>(y1,x0);
    int p3 = input.at<uchar>(y1,x1);

    float top = p0*(1-peso_x)+p1*peso_x;
    float bottom = p2*(1-peso_x)+p3*peso_x;

    return (uchar) (top*(1-peso_y)+bottom*peso_y);
    
}

void runTranslationCPU(const Mat& input, Mat& output, int dx, int dy)
{
    controlloAllocazione(input,output);
    int rows = input.rows;
    int cols = input.cols;

    for(int y=0;y<rows;y++)
    {
        for(int x=0;x<cols;x++)
        {
            int x_in = x-dx;
            int y_in = y-dy;

            if(x_in<0 || x_in>=cols || y_in<0 || y_in>=rows)
                output.at<uchar>(y,x) = 0;
            else
                output.at<uchar>(y,x) = input.at<uchar>(y_in,x_in);
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

    for(int y=0;y<rows;y++)
    {
        for(int x=0;x<cols;x++)
        {
            float dx = x-cx;
            float dy = y-cy;
            
            float x_in = (dx*coseno-dy*seno)+cx;
            float y_in = (dx*seno+dy*coseno)+cy;

            output.at<uchar>(y,x)=interpolazione_bilineare(input,x_in,y_in,rows,cols);
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

    for(int y=0;y<rows;y++)
    {
        for(int x=0;x<cols;x++)
        {
            float x_in = (x-cx)/scala + cx;
            float y_in = (y-cy)/scala+cy;
            output.at<uchar>(y,x)=interpolazione_bilineare(input,x_in,y_in,rows,cols);
        }
    }
}

