#include <stdio.h>
#include <stdlib.h>

#include "../include/cuda/info.cuh"

int main()
{
    /**
     * @brief Ejecutamos una análisis sobre el sistema para 
     * determinar los siguientes aspectos:
     *  - 1. el número de GPUsq del sistema,
     *  Por cada GPU,
     *    - 2. espacio disponible y total (en GB),
     *    - 3. identificador del sistema (entero: 0, 1, ...),
     *    - 4. número máximo de hilos por bloque,
     *    - 5. número de Streaming Multiprocessors (SM),
     *    - 6. tamaño de memoria compatida (en MB),
     *    - 7. tamaño máximo en hilo disponibles por dimensión,
     *    - 8. número máximo de bloues por cada dimensión del grid.
     */
    print_gpu_info();
    
    return 0;
}