#include <iostream>
#include <chrono> 
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "OpenCLManager.hpp"
#include "AlgoritmiCPU.hpp"
#include <fstream>
#include <filesystem>
#include <cstdlib>

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

bool verificaCorrettezza(const Mat& input, OpenCLManager& manager, Algoritmo algo)
{
    size_t dim_frame = input.total() * input.elemSize();
    size_t allineamento = manager.getByteAllineamento();
    size_t alloc_size = ((dim_frame+allineamento-1)/allineamento)*allineamento;

    // allocazione dei buffer per i test 
    void* ptr_cpu = aligned_alloc(allineamento, alloc_size);
    void* ptr_std = aligned_alloc(allineamento, alloc_size);
    void* ptr_zero = aligned_alloc(allineamento, alloc_size);

    Mat output_cpu(input.rows, input.cols, CV_8UC1, ptr_cpu);
    Mat output_gpu_std(input.rows, input.cols, CV_8UC1, ptr_std);
    Mat output_gpu_zero(input.rows, input.cols, CV_8UC1, ptr_zero); 

    runCPU(input, output_cpu, algo);
    runGPU(input, output_gpu_std, manager, algo, Modalita::GPU_STANDARD);
    runGPU(input, output_gpu_zero, manager, algo, Modalita::GPU_ZERO);
    
    Mat diff_std, diff_zero;

    // calcolo della differenza assoluta pixel per pixel tra CPU e modalità GPU  
    absdiff(output_cpu, output_gpu_std, diff_std); 
    absdiff(output_cpu, output_gpu_zero, diff_zero);
    
    // ispezione della matrice differenza per calcolare min e max dell'errore
    double max_err_std, max_err_zero;
    minMaxLoc(diff_std, NULL, &max_err_std);
    minMaxLoc(diff_zero, NULL, &max_err_zero);
    
    const double TOLLERANZA = 1.0; 
    
    bool risultato;

    if(max_err_std <= TOLLERANZA && max_err_zero <= TOLLERANZA) 
        risultato = true;
    else 
    {
        cerr << "\n[ERRORE] " << nomeAlgoritmo(algo) << " ha fallito il test!"
             << "\nErrore max tra CPU e GPU Standard: " << max_err_std 
             << " | Errore max tra CPU e GPU Zero-Copy: " << max_err_zero << endl;
        risultato=false;
    }

    free(ptr_cpu);
    free(ptr_std);
    free(ptr_zero);
    return risultato;
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
    
    VideoCapture init_cap(video_path);
    if (!init_cap.isOpened()) 
    {
        cerr << "Errore: impossibile aprire il video in " << video_path << endl;
        return -1;
    }

    int larghezza = init_cap.get(CAP_PROP_FRAME_WIDTH);
    int altezza = init_cap.get(CAP_PROP_FRAME_HEIGHT);
    init_cap.release(); 

    size_t dim_frame = larghezza * altezza; 

    // otteniamo allineamento del device richiesto e calcoliamo i bytes da allocare al multiplo più vicino  
    size_t allineamento = manager.getByteAllineamento();
    size_t alloc_size = ((dim_frame+allineamento-1)/allineamento)*allineamento;

    // allocazione memoria allineata 
    void* ptr_in = aligned_alloc(allineamento, alloc_size);
    void* ptr_out = aligned_alloc(allineamento, alloc_size);

    if (!ptr_in || !ptr_out) 
    {
        cerr << "Errore di allocazione della memoria allineata" << endl;
        if (ptr_in) free(ptr_in);
        if (ptr_out) free(ptr_out);
        return -1;
    }

    // operazione di wrapping nei cv::Mat
    Mat frame_input;
    Mat frame_grigio(altezza, larghezza, CV_8UC1, ptr_in);
    Mat frame_output(altezza, larghezza, CV_8UC1, ptr_out);

    bool primo = true;

    try{
        cout << "FASE DI VALIDAZIONE CORRETTEZZA ALGORITMI" << endl;

        VideoCapture test(video_path);

        if (!test.isOpened()) {
            cerr << "Errore: impossibile aprire il video in " << video_path << endl;
            free(ptr_in); 
            free(ptr_out);
            return -1;
        }

        test.read(frame_input);
        cvtColor(frame_input, frame_grigio, COLOR_BGR2GRAY);
        test.release();

        bool corretto = true;
        for(Algoritmo algo : algoritmi) 
        {
            cout << "Verifica " << nomeAlgoritmo(algo) << "... ";
            if(verificaCorrettezza(frame_grigio, manager, algo)) {
                cout << "OK!" << endl;
            } else {
                corretto = false;
            }
        }
        
        if(!corretto) 
        {
            free(ptr_in);
            free(ptr_out);
            return -1;
        }
        
        for(Modalita mod : modalita)
        {
            for(Algoritmo algo : algoritmi)
            {
                if (primo)
                {
                    csv<<"TEST,,,,,\n";
                    primo = false;
                }

                VideoCapture cap(video_path);
                if (!cap.isOpened()) 
                {
                    cerr << "Errore: impossibile aprire il video in " << video_path << endl;
                    free(ptr_in); 
                    free(ptr_out);
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
                
                cap.release();

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
        free(ptr_in);
        free(ptr_out);
        return -1;
    }
    
    cout << "\nBenchmark completato con successo" << endl;
    csv.close();
    free(ptr_in);
    free(ptr_out);
    return 0;
}
  