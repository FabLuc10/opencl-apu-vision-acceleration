#include <iostream>
#include <chrono> // libreria di timing ad alta precisione
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "OpenCLManager.hpp"
#include "AlgoritmiCPU.hpp"
#include <fstream>
#include <filesystem>

/*
    <chrono> si basa si basa su 3 concetti:
        - Clock: la sorgente che tiene il tempo 
        - Time point: istante bloccato 
        - Duration: la differenza tra due punti temporali

    tre tipi di orologio: 
        - std::chrono::high_resolution_clock -> orologio ad alta precisione 
        - std::chrono::steady_clock -> orologio monotonico
        - std::chrono::system_clock -> orologio di sistema 

    FPS medi = Numero di Frame / tempo totale 

*/

using namespace std;
using namespace cv;

enum class Algoritmo{SOBEL, BLUR, EROSION, DILATION, TRANSLATION, ROTATION, SCALING };
enum class Modalita{CPU_MODE, GPU_STANDARD, GPU_ZERO};

// parametri fissi per algoritmi 
const int dx = 100;
const int dy = -70;
const float grado_rotazione = 37.0f;
const float scala = 1.5f;


Algoritmo scegliAlgoritmo()
{
    cout<<"\nScegli l'algoritmo da eseguire:\n"<<"1 Sobel\n2 Blur Gaussian\n3 Erosion\n4 Dilation\n5 Translation\n6 Rotation\n7 Scaling"<<endl;
    int scelta;
    cin>>scelta;

    switch(scelta)
    {
        case 1: return Algoritmo::SOBEL; 
        case 2: return Algoritmo::BLUR;
        case 3: return Algoritmo::EROSION;
        case 4: return Algoritmo::DILATION;
        case 5: return Algoritmo::TRANSLATION;
        case 6: return Algoritmo::ROTATION;
        case 7: return Algoritmo::SCALING;
        default: return Algoritmo::BLUR;
    }
}

Modalita scegliModalita()
{
    cout<<"\nScegli la modalità di esecuzione:\n"<<"1 CPU\n2 GPU Standard\n3 GPU Zero-copy"<<endl;
    int scelta;
    cin>>scelta;

    switch(scelta)
    {
        case 1: return Modalita::CPU_MODE; 
        case 2: return Modalita::GPU_STANDARD;
        case 3: return Modalita::GPU_ZERO;
        default: return Modalita::GPU_STANDARD;
    }
}

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
    Algoritmo algoritmo_scelto = scegliAlgoritmo();
    Modalita mod_scelta = scegliModalita();

    cout<<nomeAlgoritmo(algoritmo_scelto)<<" - "<<nomeModalita(mod_scelta)<<endl;

    OpenCLManager manager;
    if(mod_scelta != Modalita::CPU_MODE) 
        manager.buildPrograms("kernels/filtri.cl","kernels/morfologia.cl","kernels/geometria.cl");

    VideoCapture cap(0);

    if(!cap.isOpened())
    {
        cerr<<"Impossibile aprire la webcam del dispositivo"<<endl;
        return -1;
    }

    ofstream csv;
    apriCSV("results/risultati_demo.csv",csv);

    Mat frame_input, frame_grigio, frame_output;

    // variabili per benchmark
    int conta_frame = 0; // per tenere traccia di quanti frame processati 
    double somma_tempo = 0.0; //variabile che memorizza il tempo accumulato per l'esecuzione degli algoritmi 
    auto inizio_frame = chrono::high_resolution_clock::now(); // time point di inizio per calcolo degli FPS medi
    const int max_frame = 30; //dimensione del campione statistico

    // crea una finestra ridimensionabile manualmente
    namedWindow("Benchmark", WINDOW_NORMAL);

    // fissiamo la dimensione della finestra 
    resizeWindow("Benchmark", 1920, 540);

    cout<<"Premere ESC per terminare"<<endl;

    bool fallback = false;

    while(cap.read(frame_input))
    {
        // trasformazione del frame a colori in scala di grigi 
        cvtColor(frame_input, frame_grigio, COLOR_BGR2GRAY);

        auto inizio = chrono::high_resolution_clock::now();

        try{
            if(mod_scelta == Modalita::CPU_MODE)
                runCPU(frame_grigio,frame_output,algoritmo_scelto);
            else 
                runGPU(frame_grigio,frame_output,manager,algoritmo_scelto,mod_scelta);
        } catch(const BufferNonAllineatoException& e)
        {
            if(!fallback)
            {
                cerr << e.what()<<"\nPassaggio a modalita' GPU Standard"<<endl; 
                fallback = true;
            }
            mod_scelta = Modalita::GPU_STANDARD; 
            runGPU(frame_grigio,frame_output,manager,algoritmo_scelto,mod_scelta);
        }
        
        auto fine = chrono::high_resolution_clock::now();

        double tempo_ms = chrono::duration<double,milli>(fine-inizio).count();

        conta_frame++;
        somma_tempo+=tempo_ms;

        if(conta_frame==max_frame)
        {
            
            double secondi = chrono::duration<double>(chrono::high_resolution_clock::now()-inizio_frame).count();
            double fps_medi = max_frame/secondi;
            double tempo_medio_ms = somma_tempo/conta_frame;

            cout << "[" << nomeAlgoritmo(algoritmo_scelto) << " | " << nomeModalita(mod_scelta) << "] "
                 << "FPS medi: " << fps_medi << " | tempo medio/frame: " << tempo_medio_ms << " ms" << endl;

            auto tempo_attuale = chrono::system_clock::now();
            time_t timestamp = chrono::system_clock::to_time_t(tempo_attuale);

            csv << nomeAlgoritmo(algoritmo_scelto) << "," << nomeModalita(mod_scelta) << ","
                << frame_grigio.cols << "x" << frame_grigio.rows << ","<< fps_medi << ","
                << tempo_medio_ms << "," 
                << put_time(localtime(&timestamp), "%Y-%m-%d %H:%M:%S") << "\n";

            csv.flush(); 

            conta_frame=0;
            somma_tempo=0;
            inizio_frame = chrono::high_resolution_clock::now();
        }

        // affianco frame di input con frame di output formando un frame di larghezza pari al doppio delle due
        Mat frame_finale;
        hconcat(frame_grigio,frame_output,frame_finale);
        
        // disegna testo sul frame finale 
        putText(frame_finale,"INPUT",Point(10,25),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255),2);
        putText(frame_finale,nomeAlgoritmo(algoritmo_scelto) + " - " + nomeModalita(mod_scelta),Point(frame_grigio.cols+10,25),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255),2);

        // creazione finestra GUI e visualizzazione dell'immagine
        imshow("Benchmark",frame_finale);

        // se viene premuto il tasto ESC si esce dal ciclo 
        if(waitKey(1)==27) break;

    }
    csv.close();
    return 0;
    
}