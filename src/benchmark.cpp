#include <iostream>
#include <chrono> 
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "OpenCLManager.hpp"
#include "AlgoritmiCPU.hpp"
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;

enum class Algoritmo{SOBEL, BLUR, EROSION, DILATION, TRANSLATION, ROTATION, SCALING };
enum class Modalita{CPU_MODE, GPU_STANDARD, GPU_ZERO};

const int dx = 100;
const int dy = -70;
const float grado_rotazione = 37.0f;
const float scala = 1.5f;

string nomeAlgoritmo(Algoritmo algo)
{
    switch(algo)
    {
        case Algoritmo::SOBEL: return "Sobel"; 
        case Algoritmo::BLUR: return "Blur";
        case Algoritmo::EROSION: return "Erosion";
        case Algoritmo::DILATION: return "Dilation";
        case Algoritmo::TRANSLATION: return "Translation";
        case Algoritmo::ROTATION: return "Rotation";
        case Algoritmo::SCALING: return "Scaling";
        default: return "";
    }
}

string nomeModalita(Modalita mod)
{
    switch(mod)
    {
        case Modalita::CPU_MODE: return "CPU";
        case Modalita::GPU_STANDARD: return "GPU Standard";
        case Modalita::GPU_ZERO: return "GPU Zero-copy";
        default: return "";
    }
}

void runCPU(const Mat& input, Mat& output, Algoritmo algo)
{
    switch(algo)
    {
        case Algoritmo::SOBEL: 
            runSobelCPU(input,output);
            break;
        case Algoritmo::BLUR:
            runBlurCPU(input,output);
            break;
        case Algoritmo::EROSION: 
            runErosionCPU(input,output);
            break;
        case Algoritmo::DILATION: 
            runDilationCPU(input,output);
            break;
        case Algoritmo::TRANSLATION: 
            runTranslationCPU(input,output,dx,dy);
            break;
        case Algoritmo::ROTATION: 
            runRotationCPU(input,output,grado_rotazione);
            break;
        case Algoritmo::SCALING: 
            runScalingCPU(input,output,scala);
            break;
    }
}

void runGPU(const Mat& input, Mat& output, OpenCLManager& manager, Algoritmo algo,Modalita mode)
{
    bool standard = mode==Modalita::GPU_STANDARD;

    switch(algo)
    {
        case Algoritmo::BLUR: 
            standard ? manager.runBlurStandard(input,output): manager.runBlurZero(input,output);
            break;
        case Algoritmo::SOBEL: 
            standard ? manager.runSobelStandard(input,output): manager.runSobelZero(input,output);
            break;
        case Algoritmo::DILATION: 
            standard ? manager.runDilationStandard(input,output): manager.runDilationZero(input,output);
            break;
        case Algoritmo::EROSION: 
            standard ? manager.runErosionStandard(input,output): manager.runErosionZero(input,output);
            break;
        case Algoritmo::TRANSLATION: 
            standard ? manager.runTranslationStandard(input,output,dx,dy): manager.runTranslationZero(input,output,dx,dy);
            break;
        case Algoritmo::ROTATION: 
            standard ? manager.runRotationStandard(input,output,grado_rotazione): manager.runRotationZero(input,output,grado_rotazione);
            break;
        case Algoritmo::SCALING: 
            standard ? manager.runScalingStandard(input,output,scala): manager.runScalingZero(input,output,scala);
            break;
    }
}

void apriCSV(const string& file_path, ofstream& csv)
{
    bool esiste = filesystem::exists(file_path);
    csv.open(file_path,ios::app);
    if(!esiste)
        csv<<"algoritmo,modalita,risoluzione,fps_medi,tempo_medio_ms,timestamp\n";
}


int main()
{
    vector<Algoritmo> algoritmi = {Algoritmo::SOBEL, Algoritmo::BLUR, Algoritmo::EROSION, 
                                              Algoritmo::DILATION, Algoritmo::TRANSLATION, 
                                              Algoritmo::ROTATION, Algoritmo::SCALING};
    vector<Modalita> modalita = { Modalita::GPU_ZERO,Modalita::GPU_STANDARD,Modalita::CPU_MODE};

    OpenCLManager manager;
    manager.buildPrograms("kernels/filtri.cl", "kernels/morfologia.cl", "kernels/geometria.cl");  

    ofstream csv;
    apriCSV("results/benchmark.csv",csv);

    string video_path="media/video_benchmark.mp4";

    Mat frame_input, frame_grigio, frame_output;

    bool primo = true;

    try{
        for(Modalita mod : modalita)
        {
            for(Algoritmo algo : algoritmi)
            {
                if (primo)
                {
                    csv<<"\nTEST\n";
                    primo = false;
                }
                VideoCapture cap(video_path);
                if (!cap.isOpened()) {
                    cerr << "Errore: impossibile aprire il video in " << video_path << endl;
                    return -1;
                }

                cout << "Test in corso: [" << nomeAlgoritmo(algo) << " - " << nomeModalita(mod) << "]... " << endl;

                int conta_frame = 0; 
                double somma_tempo = 0.0; 
                const int max_frame = 150;

                while(cap.read(frame_input) && conta_frame < max_frame)
                {
                    cvtColor(frame_input, frame_grigio, COLOR_BGR2GRAY);

                    auto inizio = chrono::high_resolution_clock::now();

                    if(mod == Modalita::CPU_MODE)
                        runCPU(frame_grigio, frame_output, algo);
                    else 
                        runGPU(frame_grigio, frame_output, manager, algo, mod);
                    
                    auto fine = chrono::high_resolution_clock::now();

                    somma_tempo += chrono::duration<double, milli>(fine - inizio).count();
                    conta_frame++;
                }

                double tempo_medio_ms = somma_tempo/conta_frame;
                double fps_puri = 1000.0/tempo_medio_ms; 

                cout << "Algoritmo finito (" << fps_puri << " FPS medi)" << endl;

                
                auto tempo_attuale = chrono::system_clock::now();
                time_t timestamp = chrono::system_clock::to_time_t(tempo_attuale);

                csv << nomeAlgoritmo(algo) << "," << nomeModalita(mod) << ","
                    << frame_grigio.cols << "x" << frame_grigio.rows << "," << fps_puri << ","
                    << tempo_medio_ms << "," 
                    << put_time(localtime(&timestamp), "%Y-%m-%d %H:%M:%S") << "\n";
                csv.flush(); 
            }
        }
    } catch(const BufferNonAllineatoException& e)
    {
        cerr << "\n[ERRORE] " << e.what() << "\nBenchmark interrotto" << endl;
        csv << "ERRORE,,,,,\n";
        csv.flush();
        csv.close();
        return -1;
    }
    
    cout << "\nBenchmark completato con successo" << endl;
    csv.close();
    return 0;
}
  