typedef struct CeldaLV lvcelda;
typedef struct CeldaIndice icelda;

typedef struct ESIndice esindice;

typedef unsigned long long int W_Hadamard;

std::fstream file;

/*
 *
 */
struct CeldaLV
{
    int clase = -1;
    int dbs_idx = -1;
    int distancia = -1;

    W_Hadamard *Codigo;

};

struct CeldaIndice
{
    int clase = -1;
    int dbs_idx = -1;
    W_Hadamard *Codigo;

    icelda *Siguiente;

};

struct ESIndice
{
    icelda *Inicio;
    icelda *Final;
};


/*
 *
 */
class ES
{
    public:

        /* constructor y destructor
        *
        */
        ES(const int potencia, const int bits, const int vecinos, const int bloque_vecinos);
        ~ES();

        /* variables generales
        *
        */

        // variables de los iteradores
        int i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0, p = 0, q = 0, r = 0, s = 0;

        // variables de configuracion
        int n_codigos = 0;
        int n_bits = 0;
        int n_celdas = 0;
        int k_vecinos = 0;

        // variable para contar los bits activos al hacer operaciones
        int total_bits_activos = 0;

        // variables para almacenar codigo y clase actuales
        W_Hadamard *Codigo;
        int Clase = -1;
        int DBS_index = -1;

        // numero de elementos en el conjunto de entrenamiento
        int NElementos = 0;

        // para la lectura de los archivos csv
        int kl = 1;

        std::string linea;
        std::string line_value;
        std::string octeto_bytes = "";

        char *linea_codigo;
        int idx_class = 0;

        // para obtener el tiempo 
        clock_t begin = 0;
        clock_t end = 0;
        clock_t total_time = 0;


    
        /* funciones y variables para el indice
        *
        */
        ESIndice * IES;

        CeldaIndice *ite_icelda = (icelda *)malloc(sizeof(icelda));
        CeldaIndice *ite_icelda2 = (icelda *)malloc(sizeof(icelda));

        void indexar();
        void iniciar_indice();
        void contar_elementos();
        void k_vecinos_cercanos();
        void crear_indice(const std::string class_file_train, const std::string vector_file_train);
        void ejecutar_indice(const std::string class_file_val, const std::string vector_file_val);

        // variables para el indice
        bool is_fill_lv = false;
        bool can_delete_lv = false;
        //bool * lv_idxs;

        //bool * lv_dst;
        int lv_ = 0;
        int ** lv_nvd;
        int  * lv_nvx;

        int lv_len = 0;
        int lv_idx_max = 0;
        int lv_max = 0;
        int lv_cls = 0;
        lvcelda *lista_vecinos;

        int lvf_idx = 0;
        lvcelda *lista_vecinos_final;

        int * I;
        int * D;

        // @1, @5, @10, hsp, hsp+
        int a01 = 0;
        int a05 = 0;
        int a10 = 0;

        int max_knnv = 0;
        int v05 = 0;
        int v07 = 0;
        int v09 = 0;

        int vhsp = 0;
        int vhsp_inf = 0;
        int vhsp_knn = 0;

        int *celda_respuesta_votados;

        void reiniciar_k_vecinos();
        void k_votaciones(int *v_);
        void eliminar_vecinos_lejanos();

        /* para salvar csv
         *
         */
        std::string  baseFile = "";
        std::string  fileName = "";
        std::string db_name   = "";

        std::stringstream ss;

        std::vector<int> clases_valores;
        std::vector<int> clases_valores_val;

        std::ifstream archivo_clases;
        std::ifstream archivo_vectores;

        std::ifstream archivo_clases_val;
        std::ifstream archivo_vectores_val;

        // limpiar memoria
        void limpiar_memoria();

};

// Constructor ...
ES::ES(const int potencia, const int bits, const int vecinos, const int bloque_vecinos)
{
    //
    n_codigos = pow(2, potencia);
    n_bits = bits;
    n_celdas = n_codigos / n_bits;
    k_vecinos = vecinos;

    //
    NElementos = 0;

    //
    Codigo  = (W_Hadamard *)calloc(n_celdas, sizeof(W_Hadamard));

    //
    //lv_len = 100; // n_codigos;
    lv_len = bloque_vecinos;
    lista_vecinos = (lvcelda *)calloc(lv_len, sizeof(lvcelda));
    lista_vecinos_final = (lvcelda *)calloc(lv_len, sizeof(lvcelda));

    //lv_dst = (bool *) calloc(n_codigos + 1, sizeof(bool));
    lv_nvd = (int **) calloc(n_codigos + 1, sizeof(int*));
    lv_nvx = (int *) calloc(n_codigos + 1, sizeof(int));
    
    I = (int *) calloc(k_vecinos, sizeof(int));
    D = (int *) calloc(k_vecinos, sizeof(int));

    for (i = 0; i < n_codigos + 1; i++)
    {
        //lv_dst[i] = false;
        lv_nvd[i] = (int *) calloc(lv_len, sizeof(int));
        lv_nvx[i] = 0;
    }

    celda_respuesta_votados = (int *)calloc(n_codigos, sizeof(int));

}

// Destructor ...
ES::~ES()
{
    //
    //std::cout << "limpiador activado";
}

void ES::crear_indice(const std::string class_file_train,
                      const std::string vector_file_train )
{
    // para leer las classes (train)
    archivo_clases.close();
    archivo_clases.clear();
    archivo_clases.open(class_file_train);

    linea.clear();

    clases_valores.clear();

    while (std::getline(archivo_clases, linea))
    {
        //std::stringstream ss(linea);
        ss.str(linea);

        line_value = "";
        while (std::getline(ss, line_value, ','))
        {
            clases_valores.push_back(atoi(line_value.c_str()));
            break;
        }

        ss.clear();
        //ss.str("");
    }


    // para leer los vectores (train)
    archivo_vectores.close();
    archivo_vectores.clear();
    archivo_vectores.open(vector_file_train);

    linea.clear();

    i = 0, k = 0;
    idx_class = 0;

    octeto_bytes = "";
    // std::string octeto_bytes_all = "";
    //std::ifstream archivo_vectores(vector_file_train);

    //printf("Leyendo los vectores del entrenamiento\n");
    while (std::getline(archivo_vectores, linea))
    {
        //std::stringstream ss(linea);
        ss.str(linea);

        i = 0;
        k = 0;
        octeto_bytes = "";
        // octeto_bytes_all = "";
        line_value = "";

        while (std::getline(ss, line_value, ','))
        {
            octeto_bytes += line_value;
            if (i % n_bits == (n_bits - 1))
            {
                // printf("%s", octeto_bytes.c_str());
                // octeto_bytes_all += octeto_bytes;
                //W_Hadamard ulli1 = std::strtoull(octeto_bytes.c_str(), &linea_codigo, 2);
                //Codigo[k] = ulli1;
                Codigo[k] = std::strtoull(octeto_bytes.c_str(), &linea_codigo, 2);

                k += 1;
                octeto_bytes = "";
            }

            i += 1;
        }

        //
        DBS_index = idx_class;
        Clase = clases_valores[idx_class];
        indexar();

        //
        idx_class += 1;
        ss.clear();
        //ss.str("");
    }

    // contar los elementos en el indice
    contar_elementos();

}

/*
 *
 */
void ES::ejecutar_indice(const std::string class_file_val, const std::string vector_file_val)
{
    // create csv
    //fileName = baseFile+"/es_"+db_name+".txt";
    file.open(baseFile+"/es_"+db_name+".txt", std::ios::out | std::ios::trunc );

    // para leer las classes (val)
    archivo_clases_val.close();
    archivo_clases_val.clear();
    archivo_clases_val.open(class_file_val);

    linea.clear();

    clases_valores.clear();

    while(std::getline(archivo_clases_val, linea))
    {
        //std::stringstream
        ss.str(linea);

        line_value = "";
        while(std::getline(ss, line_value, ','))
        {
            clases_valores.push_back(atoi(line_value.c_str()));
            break;
        }

        ss.clear();
        //ss.str("");
    }

    // re iniciar los contadores
    a01 = 0;
    a05 = 0;
    a10 = 0;

    v05 = 0;
    v07 = 0;
    v09 = 0;

    vhsp = 0;
    vhsp_inf = 0;

    // para leer los vectores (train)
    archivo_vectores_val.close();
    archivo_vectores_val.clear();
    archivo_vectores_val.open(vector_file_val);

    linea.clear();
    ss.clear();

    i = 0, k = 0, kl = 0;
    idx_class = 0;

    //printf("Leyendo los vectores de la validacion\n");
    while(std::getline(archivo_vectores_val, linea))
    {
        //std::stringstream ss(linea);
        ss.str(linea);

        i = 0;
        k = 0;
        octeto_bytes = "";
        line_value = "";

        while(std::getline(ss, line_value, ','))
        {
            octeto_bytes += line_value;
            if (i%n_bits == (n_bits - 1)) {
                //printf("%s\n", octeto_bytes.c_str());
                //W_Hadamard ulli1 = std::strtoull(octeto_bytes.c_str(), &linea_codigo, 2);
                //Codigo[k] = ulli1;
                Codigo[k] = std::strtoull(octeto_bytes.c_str(), &linea_codigo, 2);

                k += 1;
                octeto_bytes = "";
            }

            i += 1;
        }
        
        //
        DBS_index = idx_class;
        Clase = clases_valores[idx_class];
        k_vecinos_cercanos();

        // para escribir archivo csv 
        for (i = 0; i < k_vecinos; i++)
        {
            file << I[i];
        
            if (i != (k_vecinos - 1)) file << ", ";
        }
        file << "\n";

        //
        idx_class += 1;
        kl += 1;

        //if (idx_class == 10) break;

        ss.clear();
        //ss.str(std::string());
        //ss.str("");
    }

    file.close();
}

void ES::iniciar_indice()
{
    IES = (esindice *)malloc(sizeof(esindice));
    IES->Inicio = NULL;
    IES->Final = NULL;
}

void ES::indexar()
{
    if (IES->Inicio == NULL && IES->Final == NULL)
    {
        IES->Final = (icelda *)malloc(sizeof(icelda));
        IES->Inicio = IES->Final;
    }
    else
    {
        IES->Final->Siguiente = (icelda *)malloc(sizeof(icelda));
        IES->Final = IES->Final->Siguiente;
    }

    // copiar datos ... 
    IES->Final->clase = Clase;
    IES->Final->dbs_idx = DBS_index;
    IES->Final->Codigo = (W_Hadamard *)calloc(n_celdas, sizeof(W_Hadamard));

    for (k = 0; k < n_celdas; k++)
    {
        IES->Final->Codigo[k] = Codigo[k];
    }

    IES->Final->Siguiente = NULL;

    NElementos += 1;
}

void ES::contar_elementos()
{
    NElementos = 0;
    ite_icelda = IES->Inicio;
 
    while (ite_icelda != NULL)
    {
        NElementos += 1;
        ite_icelda = ite_icelda->Siguiente;
    }

    printf("Elementos indexados: %d\n", NElementos);
}

void ES::reiniciar_k_vecinos()
{

    // reiniciar la lista de los vecinos más cercanos
    is_fill_lv = false;
    lv_idx_max = 0;

    // reiniciar los votos para el kNN
    for (i = 0; i < n_codigos; i++)
    {
        celda_respuesta_votados[i] = 0;
    }
}

void ES::k_votaciones(int *v_)
{
    for (j = 0; j < n_codigos; j++)
    {
        if (max_knnv == celda_respuesta_votados[j] && j == Clase)
        {
            *v_ += 1;
            break;
        }
    }
}

void ES::eliminar_vecinos_lejanos()
{
    lv_idx_max = 0;
    lv_max = lista_vecinos[0].distancia;
    lv_cls = lista_vecinos[0].clase;

    for (j = 0; j < lv_len; j++)
    {
        if ((lv_max <  lista_vecinos[j].distancia) || 
            (lv_max == lista_vecinos[j].distancia && lv_cls < lista_vecinos[j].clase))
        //if (lv_max <  lista_vecinos[r].distancia)
        {
            lv_idx_max = j;
            lv_max = lista_vecinos[j].distancia;
            lv_cls = lista_vecinos[j].clase;
        }
    }
}

void ES::k_vecinos_cercanos()
{
    //
    reiniciar_k_vecinos();

    //
    total_bits_activos = 0;

    ite_icelda = IES->Inicio;

    for (i = 0; i < NElementos; i++)
    {
        total_bits_activos = 0;
        for (j = 0; j < n_celdas; j++)
        {
            total_bits_activos += __builtin_popcountll(Codigo[j] ^ ite_icelda->Codigo[j]);
        }

        // agregar o cambiar el mas cercano 
        can_delete_lv = false;
        if (!is_fill_lv)
        {
            lista_vecinos[lv_idx_max].clase = ite_icelda->clase;
            lista_vecinos[lv_idx_max].Codigo = ite_icelda->Codigo;
            lista_vecinos[lv_idx_max].dbs_idx = ite_icelda->dbs_idx;
            lista_vecinos[lv_idx_max].distancia = total_bits_activos;

            lv_idx_max += 1;
            if (lv_idx_max >= lv_len)
            {
                can_delete_lv = true;
            }
        }
        else if (is_fill_lv  && ((total_bits_activos <  lv_max) || 
                                 (total_bits_activos == lv_max && ite_icelda->clase < lv_cls)))
        //else if (is_fill_lv  && (total_bits_activos <  lv_max))
        {
            lista_vecinos[lv_idx_max].clase = ite_icelda->clase;
            lista_vecinos[lv_idx_max].Codigo = ite_icelda->Codigo;
            lista_vecinos[lv_idx_max].dbs_idx = ite_icelda->dbs_idx;
            lista_vecinos[lv_idx_max].distancia = total_bits_activos;

            can_delete_lv = true;
        }

        if (can_delete_lv)
        {
            eliminar_vecinos_lejanos();
            is_fill_lv = true;
        }

        // 
        ite_icelda = ite_icelda->Siguiente;
    }

    // para seleccionar los vecinos más cercanos 
    for (r = 0; r < n_codigos + 1; r++) { lv_nvx[r] = 0; }
    for (r = 0; r < lv_len; r++)
    {
        lv_nvd[lista_vecinos[r].distancia][lv_nvx[lista_vecinos[r].distancia]] = r;
        lv_nvx[lista_vecinos[r].distancia] += 1;
    }

    lvf_idx = 0;
    for (r = 0; r < n_codigos + 1; r++)
    {
        if (lv_nvx[r] == 0) continue;

        for (s = 0; s < lv_nvx[r]; s++)
        {
            lista_vecinos_final[lvf_idx].clase = lista_vecinos[lv_nvd[r][s]].clase;
            lista_vecinos_final[lvf_idx].Codigo = lista_vecinos[lv_nvd[r][s]].Codigo;
            lista_vecinos_final[lvf_idx].dbs_idx = lista_vecinos[lv_nvd[r][s]].dbs_idx;
            lista_vecinos_final[lvf_idx].distancia = lista_vecinos[lv_nvd[r][s]].distancia;

            // salvar para mostrar los datos 
            I[lvf_idx] = lista_vecinos_final[lvf_idx].dbs_idx;
            D[lvf_idx] = lista_vecinos_final[lvf_idx].distancia;

            //printf("%d", lista_vecinos_final[lvf_idx].dbs_idx);
            //printf("(%d, %d)", lista_vecinos_final[lvf_idx].dbs_idx, lista_vecinos_final[lvf_idx].distancia);

            lvf_idx += 1;

            if (lvf_idx >= k_vecinos) break; //else printf(", ");
        }

        if (lvf_idx >= k_vecinos) break;
    }    
}


void ES::limpiar_memoria()
{
    ite_icelda2 = IES->Inicio->Siguiente;

    free(IES->Inicio->Codigo);
    IES->Inicio->Codigo = NULL;
    free(IES->Inicio);

    while (ite_icelda2 != NULL)
    {
        ite_icelda = ite_icelda2;
        ite_icelda2 = ite_icelda2->Siguiente;

        free(ite_icelda->Codigo);
        ite_icelda->Codigo = NULL;
        free(ite_icelda);
    }
    
    free(ite_icelda2);
    free(IES);
}