#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS    //abilitazione delle eccezioni C++ per OpenCL 
#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <string>


class OpenCLManager {

    private:
        cl::Context context;
        cl::Device device;
        cl::CommandQueue queue;

        // creazione di un Program per ogni file .cl 
        cl::Program prog_filtri;
        cl::Program prog_morfologia;
        cl::Program prog_geometria;

        // allocazione buffer per metodo standard e zero-copy
        cl::Buffer input_gpu;
        cl::Buffer output_gpu;
        cl::Buffer temp_gpu;

        cl::Buffer input_gpu_zero;
        cl::Buffer output_gpu_zero;
        cl::Buffer temp_gpu_zero;

        size_t ultima_dim=0;
        size_t ultima_dim_zero=0;
        void* ultimo_ptr_in = nullptr;
        void* ultimo_ptr_out = nullptr;

        // kernels 
        cl::Kernel kernel_sobel;
        cl::Kernel kernel_blur_x;
        cl::Kernel kernel_blur_y;
        cl::Kernel kernel_erosion;
        cl::Kernel kernel_dilation;
        cl::Kernel kernel_translation;
        cl::Kernel kernel_rotation;
        cl::Kernel kernel_scaling;

        
        // funzione ausilaria per leggere un file .cl 
        std::string readFile(const std::string& filename);

        // funzioni ausiliarie che allocano buffer 
        void allocaBuffer(size_t dim);
        void allocaBufferZero(const cv::Mat& input, cv::Mat& output);

    public:
        OpenCLManager(); //costruttore
        ~OpenCLManager(); //distruttore 

        // metodo per compilare i file .cl a runtime 
        void buildPrograms(const std::string& path_filtri, const std::string& path_morfologia, const std::string& path_geometria);

        // Metodi per GPU standard 
        void runSobelStandard(const cv::Mat& input, cv::Mat& output);
        void runBlurStandard(const cv::Mat& input, cv::Mat& output);
        void runErosionStandard(const cv::Mat& input, cv::Mat& output);
        void runDilationStandard(const cv::Mat& input, cv::Mat& output);
        void runTranslationStandard(const cv::Mat& input, cv::Mat& output, int dx, int dy);
        void runRotationStandard(const cv::Mat& input, cv::Mat& output, float grado_rotazione);
        void runScalingStandard(const cv::Mat& input, cv::Mat& output, float scala);
        

        // metodi per GPU zero-copy 
        void runSobelZero(const cv::Mat& input, cv::Mat& output);
        void runBlurZero(const cv::Mat& input, cv::Mat& output);
        void runErosionZero(const cv::Mat& input, cv::Mat& output);
        void runDilationZero(const cv::Mat& input, cv::Mat& output);
        void runTranslationZero(const cv::Mat& input, cv::Mat& output, int dx, int dy);
        void runRotationZero(const cv::Mat& input, cv::Mat& output, float grado_rotazione);
        void runScalingZero(const cv::Mat& input, cv::Mat& output, float scala);

};