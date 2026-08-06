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
