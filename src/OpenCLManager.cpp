#include "OpenCLManager.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

OpenCLManager::OpenCLManager()
{
    try{
        // Piattaforme
        vector<cl::Platform> piattaforme;
        cl::Platform::get(&piattaforme); // popolo il vettore piattaforme con i driver trovati 
        if(piattaforme.empty()) 
            throw runtime_error("Piattaforma non trovata!");
        cl::Platform prima_piattaforma = piattaforme.front();
        cout<<"[OPENCL] Prima piattaforma trovata "<<prima_piattaforma.getInfo<CL_PLATFORM_NAME>()<<endl;

        // Device (GPU associata alla piattaforma trovata)
        vector<cl::Device> devices;
        prima_piattaforma.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if(devices.empty()) 
            throw runtime_error("Nessuna GPU trovata!");
        device = devices.front();
        cout<<"[OPENCL] GPU trovata "<<device.getInfo<CL_DEVICE_NAME>()<<endl;

        // Creazione contesto OpenCL 
        context = cl::Context(device);
        queue = cl::CommandQueue(context,device);
    } catch(const cl::Error& e)
    {
        cerr<<"Errore OpenCL nel costruttore: "<<e.what()<<" ("<<e.err()<<")"<<endl;
        exit(1);
    }
    catch(const exception& e) { 
        cerr<<"Errore: "<<e.what()<<endl;
        exit(1);
    }
}


OpenCLManager::~OpenCLManager()
{
    /*quando oggetto OpenCLManager verrà distrutto,
    context, queue e device verranno rimossi in automatico */

    cout<<"[OPENCL] risorse liberate con successo"<<endl;
}


string OpenCLManager::readFile(const string& filename)
{
    ifstream file(filename);
    if (!file.is_open())
        throw runtime_error("Impossibile aprire il file: " + filename);
    stringstream buffer;
    buffer<<file.rdbuf();
    return buffer.str();
}

void OpenCLManager::buildPrograms(const std::string& path_filtri, const std::string& path_morfologia, const std::string& path_geometria)
{
    try
    {
        cout<<"[OPENCL] caricamento e compilazione dei kernel"<<endl;

        // lettura file tramite funzione ausiliaria readFile, in caso di file non esistente verrà generata un'eccezione
        string kernel_filtri = readFile(path_filtri);
        string kernel_morfologia = readFile(path_morfologia);
        string kernel_geometria = readFile(path_geometria);

        // creazione oggetti Program, uno per ogni file .cl da compilare 
        prog_filtri = cl::Program(context,kernel_filtri);
        prog_geometria = cl::Program(context,kernel_geometria);
        prog_morfologia = cl::Program(context,kernel_morfologia);

        // compilazione dei kernel su GPU 
        prog_filtri.build({device});
        prog_geometria.build({device});
        prog_morfologia.build({device});

        cout<<"[OPENCL] kernel compilati con successo"<<endl;

    }
    catch(const cl::Error& e)
    {
        cerr<<"Errore OpenCL: "<<e.what()<<" ("<<e.err()<<")"<<endl;

        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            string build_log = prog_filtri.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            if(build_log.length()>2)
                cerr<<"LOG FILTRI\n"<<build_log<<endl;

            build_log = prog_geometria.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            if(build_log.length()>2)
                cerr<<"LOG GEOMETRIA\n"<<build_log<<endl;

            build_log = prog_morfologia.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            if(build_log.length()>2)
                cerr<<"LOG MORFOLOGIA\n"<<build_log<<endl;
        }
        exit(1);
    }
    catch(const exception& e)
    {
        cerr<<"Errore: "<<e.what()<<endl;
        exit(1);
    }
}

void OpenCLManager::allocaBuffer(size_t dim)
{
    if(dim==ultima_dim) return; 

    input_gpu = cl::Buffer(context, CL_MEM_READ_ONLY, dim);
    output_gpu = cl::Buffer(context, CL_MEM_WRITE_ONLY, dim);
    temp_gpu = cl::Buffer(context,CL_MEM_READ_WRITE,dim);
    ultima_dim = dim;
}

void OpenCLManager::runSobelStandard(const cv::Mat& input, cv::Mat& output)
{
    try
    {
        if(output.empty() || input.size()!=output.size() || input.type()!=output.type())
            output = cv::Mat(input.size(), input.type());
        
        size_t dim = input.total() * input.elemSize();
        
        // allocazione memoria per GPU 
        allocaBuffer(dim);

        // trasferimento dati da matrice input su CPU a buffer allocato su GPU 
        //CL_TRUE indica trasferimento dati bloccante
        queue.enqueueWriteBuffer(input_gpu, CL_TRUE, 0, dim, input.data);

        // estrazione kernel dall'oggetto Program 
        cl::Kernel kernel_sobel(prog_filtri, "sobel");

        int rows = input.rows;
        int cols = input.cols;
        kernel_sobel.setArg(0,input_gpu);
        kernel_sobel.setArg(1,output_gpu);
        kernel_sobel.setArg(2,rows);
        kernel_sobel.setArg(3,cols);

        // dimensione globale su cui lavoraranno i work-items
        cl::NDRange global_size(cols, rows);

        // esecuzione del kernel 
        queue.enqueueNDRangeKernel(kernel_sobel,cl::NullRange,global_size);

        // trasferimento dati da buffer GPU a CPU 
        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);

    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runSobelStandard: " << e.what() << " (" << e.err() << ")" << endl;
    }
    
}

void OpenCLManager::runBlurStandard(const cv::Mat& input, cv::Mat& output)
{
    try
    {
        if(output.empty() || input.size()!=output.size() || input.type()!=output.type())
            output = cv::Mat(input.size(), input.type());

        size_t dim = input.total() * input.elemSize();

        allocaBuffer(dim);
        
        // esecuzione primo kernel orizzontale 
        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);
        cl::Kernel kernel_blur_x(prog_filtri,"blur_x");

        int rows = input.rows;
        int cols = input.cols;
        kernel_blur_x.setArg(0,input_gpu);
        kernel_blur_x.setArg(1,temp_gpu);
        kernel_blur_x.setArg(2,rows);
        kernel_blur_x.setArg(3,cols);

        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_blur_x,cl::NullRange,global_size);

        // esecuzione secondo kernel verticale 
        cl::Kernel kernel_blur_y(prog_filtri,"blur_y");
        kernel_blur_y.setArg(0,temp_gpu);
        kernel_blur_y.setArg(1,output_gpu);
        kernel_blur_y.setArg(2,rows);
        kernel_blur_y.setArg(3,cols);

        queue.enqueueNDRangeKernel(kernel_blur_y,cl::NullRange,global_size);

        // trasferimento dati da GPU a CPU 
        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);


    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runBlurStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }
    
}

void OpenCLManager::runErosionStandard(const cv::Mat& input, cv::Mat& output)
{
    try{
        if(output.size()!=input.size() || output.empty() || input.type()!=output.type())
            output = cv::Mat(input.size(), input.type());

        size_t dim = input.total()*input.elemSize();

        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);
        
        cl::Kernel kernel_erosion(prog_morfologia,"erosion");   
        int rows = input.rows;
        int cols = input.cols;
        kernel_erosion.setArg(0,input_gpu);
        kernel_erosion.setArg(1,output_gpu);
        kernel_erosion.setArg(2,rows);
        kernel_erosion.setArg(3,cols);

        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_erosion,cl::NullRange,global_size);

        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);
    } catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runErosionStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }

}

void OpenCLManager::runDilationStandard(const cv::Mat& input, cv::Mat& output)
{
    try{
        if(output.empty() || input.size()!=output.size() || input.type()!=output.type())
            output = cv::Mat(input.size(),input.type());
        
        size_t dim = input.total()*input.elemSize();
        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);

        cl::Kernel kernel_dilation(prog_morfologia,"dilation");
        int rows = input.rows;
        int cols = input.cols;
        kernel_dilation.setArg(0,input_gpu);
        kernel_dilation.setArg(1,output_gpu);
        kernel_dilation.setArg(2,rows);
        kernel_dilation.setArg(3,cols);

        cl::NDRange global_size(cols,rows);
        
        queue.enqueueNDRangeKernel(kernel_dilation,cl::NullRange,global_size);

        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);
    } catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runDilationStandard: " << e.what() << " (" << e.err() << ")" << endl;    

    }
}

void OpenCLManager::runTranslationStandard(const cv::Mat& input, cv::Mat& output, int dx, int dy)
{
    try
    {
        if(output.empty() || input.size()!=output.size() || input.type()!=output.type())
            output = cv::Mat(input.size(),input.type());

        size_t dim = input.total()*input.elemSize();

        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);
        
        cl::Kernel kernel_translation(prog_geometria,"translation");
        int rows = input.rows;
        int cols = input.cols;
        kernel_translation.setArg(0,input_gpu);
        kernel_translation.setArg(1,output_gpu);
        kernel_translation.setArg(2,rows);
        kernel_translation.setArg(3,cols);
        kernel_translation.setArg(4,dx);
        kernel_translation.setArg(5,dy);

        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_translation,cl::NullRange,global_size);

        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runTranslationStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }
    
}