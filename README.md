# OpenCL APU Vision Acceleration

Progetto di tesi triennale in Ingegneria informatica: accelerazione di algoritmi di *computer vision* tramite **OpenCL** su GPU integrata (APU AMD), applicati a un flusso video live da webcam.
 
L'obiettivo è dimostrare **quanto e come** la programmazione parallela su GPU ottimizza algoritmi classici di elaborazione delle immagini, confrontando tre modalità di esecuzione per ciascun algoritmo:
 
- **CPU** — implementazione seriale di riferimento
- **GPU Standard** — esecuzione OpenCL con trasferimento esplicito dei buffer (host ↔ device)
- **GPU Zero-Copy** — esecuzione OpenCL con memoria condivisa host/device per eliminare il costo di trasferimento

## Modalità di esecuzione

| Modalità | Descrizione |
|---|---|
| **CPU** | Implementazione sequenziale in C++ puro, usata come baseline |
| **GPU Standard** | Esecuzione su GPU via OpenCL, con trasferimento esplicito dei buffer host↔device |
| **GPU Zero-copy** | Esecuzione su GPU via OpenCL sfruttando la memoria condivisa CPU/GPU delle APU, evitando le copie esplicite dei buffer |

## Algoritmi disponibili

- Sobel (edge detection)
- Blur Gaussian 
- Erosion / Dilation (morfologia)
- Translation / Rotation / Scaling (trasformazioni geometriche)

## Requisiti

- Compilatore C++17
- CMake ≥ 3.16
- [OpenCL](https://www.khronos.org/opencl/) (libreria + ICD del driver GPU)
- [OpenCV](https://opencv.org/)
- Una GPU con driver OpenCL funzionante (il progetto è stato sviluppato e testato su GPU integrata AMD Renoir, driver AMD ufficiali su Ubuntu 22.04)
- Una webcam accessibile dal sistema


Su Ubuntu:
```bash
sudo apt install cmake build-essential ocl-icd-opencl-dev libopencv-dev clinfo
```
Verificare che la GPU sia visibile a OpenCL con `clinfo -l` prima di compilare.

## Build

Il progetto usa una build **out-of-source**:

```bash
cmake -S . -B build
cmake --build build 
```

La cartella `kernels/` viene copiata automaticamente nella directory di build da CMake, così i file `.cl` sono raggiungibili a runtime (vengono compilati dal driver OpenCL all'avvio del programma, non in fase di build C++).

## Esecuzione

```bash
./build/opencl-apu-vision-acceleration
```

Il programma chiede a menu quale algoritmo e quale modalità eseguire, poi apre la webcam e mostra a schermo l'immagine di input affiancata al risultato elaborato in tempo reale. Premere **ESC** per terminare.
 
Ad ogni finestra di 30 frame elaborati, il programma stampa a console FPS medi e tempo medio di elaborazione per frame, e appende una riga a `results/benchmark_risultati.csv` con: algoritmo, modalità, risoluzione, FPS medi, tempo medio (ms), timestamp.

## Benchmark

Per rendere il confronto tra CPU e GPU rappresentativo, le implementazioni CPU in `AlgoritmiCPU.cpp` replicano fedelmente la stessa logica algoritmica dei kernel `.cl` (stessa maschera Sobel, stesso raggio di blur, stesso elemento strutturante, stessa formula di interpolazione bilineare) — il confronto misura quindi l'effetto della parallelizzazione a parità di algoritmo, non la differenza tra implementazioni diverse.
 
I parametri delle trasformazioni geometriche (traslazione, angolo di rotazione, fattore di scala) sono fissati in `main.cpp`, per garantire condizioni di carico identiche tra le esecuzioni ripetute.

## Struttura del progetto
 
```
.
├── include/                
│   ├── OpenCLManager.hpp   # Wrapper per contesto/coda/buffer/kernel OpenCL
│   └── AlgoritmiCPU.hpp    # Firme delle implementazioni degli algoritmi su CPU
├── src/                    
│   ├── main.cpp            # menu interattivo, acquisizione webcam, benchmark
│   ├── OpenCLManager.cpp   # Gestione GPU: setup, build kernel, metodi per runnare su GPU
│   └── AlgoritmiCPU.cpp    # Implementazione algoritmi in c++ puro
├── kernels/                
│   ├── filtri.cl           # sobel, blur
│   ├── morfologia.cl       # erosion, dilation
│   └── geometria.cl        # translation, rotation, scaling
├── results/
│   └── benchmark_risultati.csv   # Log dei benchmark
└── CMakeLists.txt
```

## Autore 

Fabrizio Lucia 