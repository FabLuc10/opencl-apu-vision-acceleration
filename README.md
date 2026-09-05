# OpenCL APU Vision Acceleration

Progetto di tesi triennale in Ingegneria informatica: accelerazione di algoritmi di *computer vision* tramite **OpenCL** su GPU integrata (APU AMD), applicati a un flusso video live da webcam.
 
L'obiettivo è dimostrare **quanto e come** la programmazione parallela su GPU ottimizza algoritmi classici di elaborazione delle immagini, confrontando tre modalità di esecuzione per ciascun algoritmo: **CPU**, **GPU Standard** e **GPU Zero-Copy**.
 
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
- Una GPU con driver OpenCL (il progetto è stato sviluppato e testato su GPU integrata AMD Renoir, driver AMD ufficiali su Ubuntu 22.04)
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

Il progetto produce **due eseguibili distinti**, con scopi diversi.

### Demo interattiva (webcam live)

```bash
./build/opencl_apu_vision_acceleration
```
 
Il programma chiede a menu quale algoritmo e quale modalità eseguire, poi apre la webcam e mostra a schermo l'immagine di input affiancata al risultato elaborato in tempo reale. Premere **ESC** per terminare.
 
Ad ogni finestra di 30 frame elaborati, il programma stampa a console FPS medi e tempo medio di elaborazione per frame, e appende una riga a `results/risultati_demo.csv` con: algoritmo, modalità, risoluzione, FPS medi, tempo medio (ms), timestamp.

Per garantire la corretta esecuzione della modalità Zero-copy, i buffer delle immagini vengono allocati esplicitamente in C++ (tramite `aligned_alloc`) rispettando il requisito di allineamento richiesto dal device OpenCL.

### Benchmark automatico con video precaricato Full HD

```bash
./build/benchmark
```

Prima di avviare il benchmark, il programma esegue una **fase di validazione** per accertarsi che i risultati prodotti dai kernel GPU siano coerenti con quelli prodotti dalla CPU. 

Successivamente, esegue **tutti gli algoritmi in tutte le modalità** (GPU Zero-copy, GPU Standard, CPU) in sequenza su un video precaricato (`media/video_benchmark.mp4`), elaborando un numero fisso di frame per ciascuna combinazione. A differenza della demo interattiva, non è limitato dal frame rate di acquisizione della webcam: misura il tempo di elaborazione puro, condizione necessaria per un confronto statisticamente corretto, e salva i risultati in `results/benchmark.csv`. Anche qui, la memoria viene pre-allineata dinamicamente per garantire il pieno supporto Zero-copy.

> Il video utilizzato per il benchmark è un estratto di *Big Buck Bunny* (© Blender Foundation, 2008), distribuito con licenza [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/).

## Benchmark

Per rendere il confronto tra CPU e GPU rappresentativo, le implementazioni CPU in `AlgoritmiCPU.cpp` replicano fedelmente la stessa logica algoritmica dei kernel `.cl` (stessa maschera Sobel, stesso raggio di blur, stesso elemento strutturante per le operazioni morfologiche e stessa formula di interpolazione bilineare).
Il confronto misura l'effetto della parallelizzazione a parità di algoritmo, non la differenza tra implementazioni diverse.
 
I parametri delle trasformazioni geometriche (traslazione, angolo di rotazione e fattore di scala) sono fissati in `main.cpp` e `benchmark.cpp`, per garantire condizioni di carico identiche tra le esecuzioni ripetute.

## Struttura del progetto
 
```
 .
 ├── include/                
 │   ├── OpenCLManager.hpp   # wrapper per contesto/coda/buffer/kernel OpenCL
 │   └── AlgoritmiCPU.hpp    # firme delle implementazioni degli algoritmi su CPU
 ├── src/                    
 │   ├── main.cpp            # menu interattivo, acquisizione webcam, benchmark
 │   ├── benchmark.cpp       # benchmark su video FUll HD
 │   ├── OpenCLManager.cpp   # gestione GPU: setup, build kernel, metodi per runnare su GPU
 │   └── AlgoritmiCPU.cpp    # implementazione algoritmi in c++ puro
 ├── kernels/                
 │   ├── filtri.cl           # sobel, blur
 │   ├── morfologia.cl       # erosion, dilation
 │   └── geometria.cl        # translation, rotation, scaling
 ├── media/
 │   └── video_benchmark.mp4   # video di riferimento usato da benchmark.cpp per raccogliere dati statistici
 ├── results/
 │   ├── risultati_demo.csv          # log della demo interattiva
 │   └── benchmark.csv   # log del benchmark automatico
 └── CMakeLists.txt
```

## Autore 

Fabrizio Lucia 
