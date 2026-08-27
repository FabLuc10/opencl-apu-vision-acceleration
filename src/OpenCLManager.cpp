#include "OpenCLManager.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdint>

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

        // informazioni su allineamento del device
        byte_allineamento = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>()/8;
    
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

        // compilazione dei kernel su GPU con ottimizzazioni del compilatore
        const char* cl_options = "-cl-fast-relaxed-math -cl-mad-enable";
        prog_filtri.build({device}, cl_options);
        prog_geometria.build({device}, cl_options);
        prog_morfologia.build({device}, cl_options);

        // estrazione dei kernels
        kernel_blur_x = cl::Kernel(prog_filtri,"blur_x");
        kernel_blur_y = cl::Kernel(prog_filtri,"blur_y");
        kernel_sobel = cl::Kernel(prog_filtri,"sobel");
        kernel_dilation = cl::Kernel(prog_morfologia,"dilation");
        kernel_erosion = cl::Kernel(prog_morfologia,"erosion");
        kernel_translation = cl::Kernel(prog_geometria,"translation");
        kernel_rotation = cl::Kernel(prog_geometria,"rotation");
        kernel_scaling = cl::Kernel(prog_geometria,"scaling");

        cout<<"[OPENCL] kernel compilati ed estratti con successo"<<endl;

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

bool OpenCLManager::verificaAllineamento(const cv::Mat& mat)
{
    return (reinterpret_cast<uintptr_t>(mat.data)%byte_allineamento) == 0;
}

void OpenCLManager::allocaBuffer(size_t dim)
{
    if(dim==ultima_dim) return; 

    input_gpu = cl::Buffer(context, CL_MEM_READ_ONLY, dim);
    output_gpu = cl::Buffer(context, CL_MEM_WRITE_ONLY, dim);
    
    // buffer temporaneo allocato su GPU 
    temp_gpu = cl::Buffer(context,CL_MEM_READ_WRITE,dim);
    ultima_dim = dim;
}

void OpenCLManager::allocaBufferZero(const cv::Mat& input, cv::Mat& output)
{
    size_t dim = input.total() * input.elemSize();

    if(dim!=ultima_dim_zero || input.data!=ultimo_ptr_in || output.data!=ultimo_ptr_out)
    {
        // il flag CL_MEM_USE_HOST_PTR permette di creare un buffer che punta a uno spazio di memoria già allocato su CPU 
        input_gpu_zero = cl::Buffer(context,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,dim,input.data);
        output_gpu_zero = cl::Buffer(context,CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR,dim,output.data);

        // il buffer temporaneo allocato su GPU 
        temp_gpu_zero = cl::Buffer(context,CL_MEM_READ_WRITE,dim);

        ultima_dim_zero = dim;
        ultimo_ptr_in = input.data;
        ultimo_ptr_out = output.data;
    }
}

void OpenCLManager::controlloAllocazione(const cv::Mat& input, cv::Mat& output)
{
    if(output.empty() || input.size() != output.size() || input.type() != output.type()) {
        output = cv::Mat(input.size(), input.type());
    }
}


// METODI STANDARD 

void OpenCLManager::runSobelStandard(const cv::Mat& input, cv::Mat& output)
{
    try
    {
        controlloAllocazione(input,output);
        
        size_t dim = input.total() * input.elemSize();
        
        // allocazione memoria per GPU 
        allocaBuffer(dim);

        // trasferimento dati da matrice input su CPU a buffer allocato su GPU 
        //CL_TRUE indica trasferimento dati bloccante
        queue.enqueueWriteBuffer(input_gpu, CL_TRUE, 0, dim, input.data);

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

        // garantisce che la coda OpenCL sia completamente vuota prima di passare al frame successivo
        queue.finish();

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
        controlloAllocazione(input,output);

        size_t dim = input.total() * input.elemSize();

        allocaBuffer(dim);
        
        // esecuzione primo kernel orizzontale 
        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);

        int rows = input.rows;
        int cols = input.cols;

        const int R = 10;

        // grandezza work-group orizzontale 1D 256x1
        size_t local_cols = 256;
        size_t local_rows = 1;

        // arrotondamento della di global size affinché sia multiplo di local size 
        size_t global_cols = ((cols+local_cols-1)/local_cols)*local_cols;

        // dimensione globale su cui lavoraranno i work-items
        cl::NDRange global_size(global_cols, rows);

        // dimensione local 
        cl::NDRange local_size(local_cols,local_rows);

        // bytes da allocare sono: dimensione locale + vicini del work-group che servono all'algoritmo (2*R)
        size_t bytes = (local_cols+2*R)*sizeof(uchar);

        kernel_blur_x.setArg(0,input_gpu);
        kernel_blur_x.setArg(1,temp_gpu);
        kernel_blur_x.setArg(2,rows);
        kernel_blur_x.setArg(3,cols);
        kernel_blur_x.setArg(4, cl::Local(bytes));

        queue.enqueueNDRangeKernel(kernel_blur_x,cl::NullRange,global_size,local_size);

        // grandezza work-group verticale 1x256
        local_cols = 1;
        local_rows = 256;

        // arrotondamento della di global size affinché sia multiplo di local size 
        size_t global_rows = ((rows+local_rows-1)/local_rows)*local_rows;

        // dimensione globale su cui lavoraranno i work-items
        global_size = cl::NDRange(cols, global_rows);

        // dimensione local 
        local_size = cl::NDRange(local_cols,local_rows);

        // bytes da allocare sono: dimensione locale + vicini del work-group che servono all'algoritmo (2*R)
        bytes = (local_rows+2*R)*sizeof(uchar);

        // esecuzione secondo kernel verticale 
        kernel_blur_y.setArg(0,temp_gpu);
        kernel_blur_y.setArg(1,output_gpu);
        kernel_blur_y.setArg(2,rows);
        kernel_blur_y.setArg(3,cols);
        kernel_blur_y.setArg(4,cl::Local(bytes));

        queue.enqueueNDRangeKernel(kernel_blur_y,cl::NullRange,global_size,local_size);

        // trasferimento dati da GPU a CPU 
        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);

        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runBlurStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }
    
}

void OpenCLManager::runErosionStandard(const cv::Mat& input, cv::Mat& output)
{
    try{
        controlloAllocazione(input,output);

        size_t dim = input.total()*input.elemSize();

        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);
        
        int rows = input.rows;
        int cols = input.cols;
        kernel_erosion.setArg(0,input_gpu);
        kernel_erosion.setArg(1,output_gpu);
        kernel_erosion.setArg(2,rows);
        kernel_erosion.setArg(3,cols);

        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_erosion,cl::NullRange,global_size);

        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);

        queue.finish();
    } catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runErosionStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }

}

void OpenCLManager::runDilationStandard(const cv::Mat& input, cv::Mat& output)
{
    try{
        controlloAllocazione(input,output);
        
        size_t dim = input.total()*input.elemSize();
        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);

        int rows = input.rows;
        int cols = input.cols;
        kernel_dilation.setArg(0,input_gpu);
        kernel_dilation.setArg(1,output_gpu);
        kernel_dilation.setArg(2,rows);
        kernel_dilation.setArg(3,cols);

        cl::NDRange global_size(cols,rows);
        
        queue.enqueueNDRangeKernel(kernel_dilation,cl::NullRange,global_size);

        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);

        queue.finish();
    } catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runDilationStandard: " << e.what() << " (" << e.err() << ")" << endl;    

    }
}

void OpenCLManager::runTranslationStandard(const cv::Mat& input, cv::Mat& output, int dx, int dy)
{
    try
    {
        controlloAllocazione(input,output);

        size_t dim = input.total()*input.elemSize();

        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);
        
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

        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runTranslationStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }
    
}

void OpenCLManager::runRotationStandard(const cv::Mat& input, cv::Mat& output, float grado_rotazione)
{
    try
    {
        controlloAllocazione(input,output);

        size_t dim = input.total()*input.elemSize();

        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);
        
        int rows = input.rows;
        int cols = input.cols;
        kernel_rotation.setArg(0,input_gpu);
        kernel_rotation.setArg(1,output_gpu);
        kernel_rotation.setArg(2,rows);
        kernel_rotation.setArg(3,cols);
        kernel_rotation.setArg(4,grado_rotazione);
        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_rotation,cl::NullRange,global_size);

        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);
        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runRotationStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }
}


void OpenCLManager::runScalingStandard(const cv::Mat& input, cv::Mat& output, float scala)
{
    try
    {
        controlloAllocazione(input,output);

        size_t dim = input.total()*input.elemSize();

        allocaBuffer(dim);

        queue.enqueueWriteBuffer(input_gpu,CL_TRUE,0,dim,input.data);
        
        int rows = input.rows;
        int cols = input.cols;
        kernel_scaling.setArg(0,input_gpu);
        kernel_scaling.setArg(1,output_gpu);
        kernel_scaling.setArg(2,rows);
        kernel_scaling.setArg(3,cols);
        kernel_scaling.setArg(4,scala);
        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_scaling,cl::NullRange,global_size);

        queue.enqueueReadBuffer(output_gpu,CL_TRUE,0,dim,output.data);

        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runScalingStandard: " << e.what() << " (" << e.err() << ")" << endl;    
    }
}


// METODI ZERO-COPY: si fa in modo che CPU e GPU non mantengano due copie della stessa immagine su cui lavorare

void OpenCLManager::runSobelZero(const cv::Mat& input, cv::Mat& output)
{
    controlloAllocazione(input,output);

    // se allineamento in memoria richiesto dal device non rispettato, lanciamo eccezione
    if (!verificaAllineamento(input) || !verificaAllineamento(output))
        throw BufferNonAllineatoException("runSobelZero: buffer non allineato al requisito del device (" +
            to_string(byte_allineamento) + " byte)");
   
    try{
        allocaBufferZero(input,output);

        size_t dim = input.total()*input.elemSize();

        /*
            pattern per sincronizzazione zero-copy tra GPU e CPU
            l'operazione di Map con CL_MAP_WRITE_INVALIDATE_REGION serve alla CPU a prendere il controllo del buffer
            e a invalidare la cache GPU, risparmiando il tempo dovuto al flush delle cache GPU verso RAM.
            l'Unmap forza la CPU a fare il flush delle cache verso la RAM passando il controllo del buffer alla GPU 
            che andrà a leggere dalla RAM dati sicuramente aggiornati   
        */
        void* ptr_in = queue.enqueueMapBuffer(input_gpu_zero,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,dim);
        queue.enqueueUnmapMemObject(input_gpu_zero, ptr_in);

        int rows = input.rows;
        int cols = input.cols;

        kernel_sobel.setArg(0,input_gpu_zero);
        kernel_sobel.setArg(1,output_gpu_zero);
        kernel_sobel.setArg(2,rows);
        kernel_sobel.setArg(3,cols);

        cl::NDRange global_size(cols,rows);
        queue.enqueueNDRangeKernel(kernel_sobel,cl::NullRange,global_size);

        /*
            Map con il flag CL_MAP_READ permette il flush delle cache GPU verso la RAM e invalida le cache CPU in modo tale 
            che la CPU sia costretta ad andare a leggere i dati aggiornati dalla GPU su RAM.
            L'Unmap passa il controllo del buffer alla GPU
        */
        void* ptr_out = queue.enqueueMapBuffer(output_gpu_zero, CL_TRUE, CL_MAP_READ, 0, dim);
        queue.enqueueUnmapMemObject(output_gpu_zero, ptr_out);

        queue.finish();

    } catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runSobelZero: " << e.what() << " (" << e.err() << ")" << endl;
    }

}

void OpenCLManager::runBlurZero(const cv::Mat& input, cv::Mat& output)
{
    controlloAllocazione(input,output);

    // se allineamento in memoria richiesto dal device non rispettato, lanciamo eccezione
    if (!verificaAllineamento(input) || !verificaAllineamento(output))
        throw BufferNonAllineatoException("runBlurZero: buffer non allineato al requisito del device (" +
            to_string(byte_allineamento) + " byte)");
    try
    {  
        allocaBufferZero(input,  output);

        size_t dim = input.total()*input.elemSize();

        void* ptr_in = queue.enqueueMapBuffer(input_gpu_zero,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,dim);
        queue.enqueueUnmapMemObject(input_gpu_zero, ptr_in);

        int rows = input.rows;
        int cols = input.cols;
        const int R = 10;

        size_t local_cols = 256;
        size_t local_rows = 1;

        size_t global_cols = ((cols+local_cols-1)/local_cols)*local_cols;

        cl::NDRange global_size(global_cols, rows);
        cl::NDRange local_size(local_cols, local_rows);

        size_t bytes = (local_cols+2*R)*sizeof(uchar);

        // esecuzione primo kernel
        kernel_blur_x.setArg(0,input_gpu_zero);
        kernel_blur_x.setArg(1,temp_gpu_zero);
        kernel_blur_x.setArg(2,rows);
        kernel_blur_x.setArg(3,cols);
        kernel_blur_x.setArg(4,cl::Local(bytes));
        queue.enqueueNDRangeKernel(kernel_blur_x, cl::NullRange, global_size,local_size);

        local_cols = 1;
        local_rows = 256;

        size_t global_rows = ((rows+local_rows-1)/local_rows)*local_rows;

        global_size = cl::NDRange(cols, global_rows);
        local_size = cl::NDRange(local_cols,local_rows);

        bytes = (local_rows+2*R)*sizeof(uchar);

        // esecuzione secondo kernel
        kernel_blur_y.setArg(0, temp_gpu_zero);
        kernel_blur_y.setArg(1, output_gpu_zero);
        kernel_blur_y.setArg(2, rows);
        kernel_blur_y.setArg(3, cols);
        kernel_blur_y.setArg(4,cl::Local(bytes));

        queue.enqueueNDRangeKernel(kernel_blur_y, cl::NullRange, global_size,local_size);

        void* ptr_out = queue.enqueueMapBuffer(output_gpu_zero, CL_TRUE, CL_MAP_READ, 0, dim);
        queue.enqueueUnmapMemObject(output_gpu_zero, ptr_out);

        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runBlurZero: " << e.what() << " (" << e.err() << ")" << endl;
    }
    
}

void OpenCLManager::runErosionZero(const cv::Mat& input, cv::Mat& output)
{
    controlloAllocazione(input,output);

    // se allineamento in memoria richiesto dal device non rispettato, lanciamo eccezione
    if (!verificaAllineamento(input) || !verificaAllineamento(output))
        throw BufferNonAllineatoException("runErosionZero: buffer non allineato al requisito del device (" +
            to_string(byte_allineamento) + " byte)");

    try{        
        allocaBufferZero(input,output);

        size_t dim = input.total()*input.elemSize();
        void* ptr_in = queue.enqueueMapBuffer(input_gpu_zero,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,dim);
        queue.enqueueUnmapMemObject(input_gpu_zero, ptr_in);

        int rows = input.rows;
        int cols = input.cols;

        kernel_erosion.setArg(0,input_gpu_zero);
        kernel_erosion.setArg(1,output_gpu_zero);
        kernel_erosion.setArg(2,rows);
        kernel_erosion.setArg(3,cols);

        cl::NDRange global_size(cols,rows);
        queue.enqueueNDRangeKernel(kernel_erosion,cl::NullRange,global_size);

        void* ptr_out = queue.enqueueMapBuffer(output_gpu_zero, CL_TRUE, CL_MAP_READ, 0, dim);
        queue.enqueueUnmapMemObject(output_gpu_zero, ptr_out);

        queue.finish();

    } catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runErosionZero: " << e.what() << " (" << e.err() << ")" << endl;
    }
    
}
void OpenCLManager::runDilationZero(const cv::Mat& input, cv::Mat& output)
{
    controlloAllocazione(input,output);

    // se allineamento in memoria richiesto dal device non rispettato, lanciamo eccezione
    if (!verificaAllineamento(input) || !verificaAllineamento(output))
        throw BufferNonAllineatoException("runDilationZero: buffer non allineato al requisito del device (" +
            to_string(byte_allineamento) + " byte)");

    try{
        allocaBufferZero(input,output);

        size_t dim = input.total()*input.elemSize();
        void* ptr_in = queue.enqueueMapBuffer(input_gpu_zero,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,dim);
        queue.enqueueUnmapMemObject(input_gpu_zero, ptr_in);

        int rows = input.rows;
        int cols = input.cols;

        kernel_dilation.setArg(0,input_gpu_zero);
        kernel_dilation.setArg(1,output_gpu_zero);
        kernel_dilation.setArg(2,rows);
        kernel_dilation.setArg(3,cols);

        cl::NDRange global_size(cols,rows);
        
        queue.enqueueNDRangeKernel(kernel_dilation,cl::NullRange,global_size);

        void* ptr_out = queue.enqueueMapBuffer(output_gpu_zero, CL_TRUE, CL_MAP_READ, 0, dim);
        queue.enqueueUnmapMemObject(output_gpu_zero, ptr_out);

        queue.finish();

    } catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runDilationZero: " << e.what() << " (" << e.err() << ")" << endl;
    }
}
void OpenCLManager::runTranslationZero(const cv::Mat& input, cv::Mat& output, int dx, int dy)
{
    controlloAllocazione(input,output);

    // se allineamento in memoria richiesto dal device non rispettato, lanciamo eccezione
    if (!verificaAllineamento(input) || !verificaAllineamento(output))
        throw BufferNonAllineatoException("runTranslationZero: buffer non allineato al requisito del device (" +
            to_string(byte_allineamento) + " byte)");
            
    try{
        allocaBufferZero(input,output);

        size_t dim = input.total()*input.elemSize();
        void* ptr_in = queue.enqueueMapBuffer(input_gpu_zero,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,dim);
        queue.enqueueUnmapMemObject(input_gpu_zero, ptr_in);

        int rows = input.rows;
        int cols = input.cols;
        
        kernel_translation.setArg(0,input_gpu_zero);
        kernel_translation.setArg(1,output_gpu_zero);
        kernel_translation.setArg(2,rows);
        kernel_translation.setArg(3,cols);
        kernel_translation.setArg(4,dx); 
        kernel_translation.setArg(5,dy);

        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_translation,cl::NullRange,global_size);

        void* ptr_out = queue.enqueueMapBuffer(output_gpu_zero, CL_TRUE, CL_MAP_READ, 0, dim);
        queue.enqueueUnmapMemObject(output_gpu_zero, ptr_out);
        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runTranslationZero: " << e.what() << " (" << e.err() << ")" << endl;    
    }
}

void OpenCLManager::runRotationZero(const cv::Mat& input, cv::Mat& output, float grado_rotazione)
{
    controlloAllocazione(input,output);

    // se allineamento in memoria richiesto dal device non rispettato, lanciamo eccezione
    if (!verificaAllineamento(input) || !verificaAllineamento(output))
        throw BufferNonAllineatoException("runRotationZero: buffer non allineato al requisito del device (" +
            to_string(byte_allineamento) + " byte)");
            
    try{
        allocaBufferZero(input,output);

        size_t dim = input.total()*input.elemSize();
        void* ptr_in = queue.enqueueMapBuffer(input_gpu_zero,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,dim);
        queue.enqueueUnmapMemObject(input_gpu_zero, ptr_in);
        
        int rows = input.rows;
        int cols = input.cols;
        
        kernel_rotation.setArg(0,input_gpu_zero);
        kernel_rotation.setArg(1,output_gpu_zero);
        kernel_rotation.setArg(2,rows);
        kernel_rotation.setArg(3,cols);
        kernel_rotation.setArg(4,grado_rotazione); 

        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_rotation,cl::NullRange,global_size);

        void* ptr_out = queue.enqueueMapBuffer(output_gpu_zero, CL_TRUE, CL_MAP_READ, 0, dim);
        queue.enqueueUnmapMemObject(output_gpu_zero, ptr_out);

        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runRotationZero: " << e.what() << " (" << e.err() << ")" << endl;    
    }
}

void OpenCLManager::runScalingZero(const cv::Mat& input, cv::Mat& output, float scala)
{
    controlloAllocazione(input,output);

    // se allineamento in memoria richiesto dal device non rispettato, lanciamo eccezione
    if (!verificaAllineamento(input) || !verificaAllineamento(output))
        throw BufferNonAllineatoException("runScalingZero: buffer non allineato al requisito del device (" +
            to_string(byte_allineamento) + " byte)");
            
    try{
        allocaBufferZero(input,output);

        size_t dim = input.total()*input.elemSize();
        void* ptr_in = queue.enqueueMapBuffer(input_gpu_zero,CL_TRUE,CL_MAP_WRITE_INVALIDATE_REGION,0,dim);
        queue.enqueueUnmapMemObject(input_gpu_zero, ptr_in);
        
        int rows = input.rows;
        int cols = input.cols;
        
        kernel_scaling.setArg(0,input_gpu_zero);
        kernel_scaling.setArg(1,output_gpu_zero);
        kernel_scaling.setArg(2,rows);
        kernel_scaling.setArg(3,cols);
        kernel_scaling.setArg(4,scala); 

        cl::NDRange global_size(cols,rows);

        queue.enqueueNDRangeKernel(kernel_scaling,cl::NullRange,global_size);

        void* ptr_out = queue.enqueueMapBuffer(output_gpu_zero, CL_TRUE, CL_MAP_READ, 0, dim);
        queue.enqueueUnmapMemObject(output_gpu_zero, ptr_out);

        queue.finish();
    }
    catch(const cl::Error& e)
    {
        cerr << "Errore OpenCL in runScalingZero: " << e.what() << " (" << e.err() << ")" << endl;    
    }
}
