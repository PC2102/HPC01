/*********************************************************
* Fecha: 23-05-2022
* Autor: Pedro Cardenas
* Materia: HPC-1
* Tema: Parcial 3 HPC
* ******************************************************/

#include "extraer.h"
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>

/* Primer funcion miembro: Lectura de fichero csv.
* Se presenta como un vector de vectores del
* tipo string.
* La idea es leer linea por linea y almacenar
* cada una en un vector de vectores del
* tipo string. */

std::vector<std::vector<std::string>> Extraer::ReadCSV(){
/* Abrir el fichero para lectura solamente */
std::fstream Fichero(setDatos);

/* Vector de vectores tipo string a entregar por
 * parte de la funcion */
std::vector<std::vector<std::string>> datosString;

/* Se itera a traves de cada linea, y se divide
 * el contenido dado por el separador( argumento
 * de entrada) provisto por el constructor */

std::string linea = ""; // Almacenar cada linea
while(getline(Fichero, linea)){
   /* Se crea un vector para almacenar la fila */
    std::vector<std::string> vectorFila;

   /* Se separa segun el delimitador */
    boost::algorithm::split(vectorFila,
                            linea,
                            boost::is_any_of(delimitador));
    datosString.push_back(vectorFila);
}

/* Se cierra el fichero .csv */
Fichero.close();

/* Se retorna el vector de
 * vectores del tipo string */
return datosString;
}

/* Se implementa la segunda funcion miembro
* la cual tiene como mision transformar el
* vector de vectores del tipo String, en
* una matrix Eigen. La idea es simular un
* objeto DATAFRAME de pandas para poder
* manipular los datos */

Eigen::MatrixXd Extraer::CSVToEigen(
    std::vector<std::vector<std::string>>  SETdatos,
    int filas, int columnas){
/* Se hace la pregunta si tiene cabezera p no
 * el vector de vectores del tipo String.
 * Si tiene cabecera, se debe eliminar */
if(header == true){
    filas = filas - 1;
}

/* Se itera sobre cada registro del fichero,
 * a la vez que se almacena en una matrixXd,
 * de dimension filas por columnas. Principalmente,
 * se almacenara Strings (porque llega un vector de
 * vectores del tipo String. La idea es
 * hacer un casting de String a float. */
Eigen::MatrixXd MatrizDF(columnas, filas);
for (int i = 0; i < filas; i++){
    for(int j = 0; j < columnas; j++){
        MatrizDF(j, i) = atof(SETdatos[i][j].c_str());
    }
}

/* Se transpone la matriz, dado que viene
 * por columnas por filas, para retornarla */
return MatrizDF.transpose();
}

/* Funcion para calcular el promedio
* En C++ la herencia del tipo de dato
* no es directa (sobre todo si es a partir
* de funciones dadas por otras interfaces/clases/
* biblioteclas: EIGEN, shrkml, etc...). Entonces
* se declara el tipo en una expresion "decltype"
* con el fin de tener seguridad de que tipo de dato
* retornara la funcion */
// En caso de no saber que dato encontrar usar auto y decltype (declarative type)

auto Extraer::Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean()){
return datos.colwise().mean();
}

/* Funcion para calcular la Desviacion Estandar */
/* Para implementar la desviacion estandar
* datosEscalados = xi - x.promedio */

auto Extraer::DesvStandard(Eigen::MatrixXd datos) -> decltype((datos.array().square().colwise().sum() / (datos.rows()-1)).sqrt()){
return (datos.array().square().colwise().sum() / (datos.rows()-1)).sqrt();
}

/* A continuacion se procede a implementar la funcion
* de normalizacion. La idea fundamental es que los datos
* presenten una cercana aproximacion al promedio, evitando
* los valores cuyas magnitudes son muy altas, o muy bajas
* (por ejemplo los outliers: valores atipicos) */

Eigen::MatrixXd Extraer::Normalizador(Eigen::MatrixXd datos){
Eigen::MatrixXd datosEscalados = datos.rowwise() -  Promedio(datos);
Eigen::MatrixXd MatrixNormal = datosEscalados.array().rowwise()/DesvStandard(datosEscalados);

/* Se retorna la matriz normalizada */
return MatrixNormal;
}


/* Para los algoritmos y/o modelos de Machine Learning
* se necesita dividir los datos en dos grupos.
* El primer grupo es del Entrenamiento: Se recomienda
* que sea aproximadamente el 80% de los datos. El
* segundo grupo es para Prueba: Sera el resto, es decir
* el 20%. La idea es crear una funcion que permita
* dividir los datos en los grupos de entrenamiento y prueba
* de forma automatica. Se requiere que la eleccion de los
* registros para cada grupo sea aleatoria. Esto garantiza
* que el resultado del modelo de ML presente una aceptable
* precision. */

/* A continuacion la funcion de division tomara el porcentaje superior
* de la matriz dada, para entrenamiento. La parte restante de la matriz
* dada para prueba. La funcion devolvera una tupla de 4 matrices
* dinamicas, variables independientes: "Entrenamiento" y "Prueba",
* variables independientes:  Entrenamiento y Prueba. Al utilizar la
* funcion en el principal, se debe desempaquetar la tupla, para
* obtener los cuatro conjuntos de datos. */

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Extraer::TrainTestSplit( Eigen::MatrixXd DatosNorm, float sizeTrain){
/* Se crea una variable para obtener el numero de filas totales */
int filasTotales = DatosNorm.rows();

/* Se crea una variable para otener el numero de filas de entrenamiento */
int filasTrain = round(filasTotales * sizeTrain);

/* Se crea una variable para otener el numero de filas de prueba */
int filasTest = filasTotales - filasTrain;

/* Se crea la matriz de entrenamiento: "Parte superior" de la matriz de entrada */
Eigen::MatrixXd Train = DatosNorm.topRows(filasTrain);

/* Del conjunto de entrenamiento y para este caso en especial (Dataset winedata),
 * todas las columnas de la izquierda son las variables independientes (Features)
 * y la ultima columna de la derecha representa la variable dependiente.
 * A continuacion, se declara el conjunto de entrenamiento de las variables
 * independientes "X". */
Eigen::MatrixXd X_train = Train.leftCols(DatosNorm.cols()-1);

/* A continuacion se declara el conjunto de entrenamientos de las variables
 * dependientes "y". */
Eigen::MatrixXd y_train = Train.rightCols(1);

/* A continuacion se declara el grupo de conjunto de datos para prueba */
Eigen::MatrixXd Test = DatosNorm.bottomRows(filasTest);

/* A continuacion se declara el conjunto de prueba de las variables
 * independientes "X", */
Eigen::MatrixXd X_test = Test.leftCols(DatosNorm.cols()-1);

/* A continuacion se declara el conjunto de prueba de las variables
 * dependientes "X", */
Eigen::MatrixXd y_test = Test.rightCols(1);

/* Finalmente se devuelve la tupla empaquetada. */
return std::make_tuple(X_train, y_train, X_test, y_test);
}

/* A continuacion se desarrollan dos nuevas funciones para
* convertir de fichero a vector, y pasar de una matriz
* a fichero. La idea principal es almacenar los valores
* parciales en ficheros por motivos de seguridad, control
* y seguimiento de la ejecucion del algoritmo de regresion
* lineal. */

/* Funcion para exportar valores de un fichero a un vector */

/* La funcion tipo void recibe un vector que contendra los valores¨
* del archivo dado. */
/* FtoV (File to Vector / Fichero a Vector) y Fname (File name) */
void Extraer::FiletoVector(std::vector<float> dataVector, std::string fileName){
/* Se crea un buffer (bus de memoria temporal) como objeto que
 * contiene la data de un fichero */
std::ofstream bufferFichero(fileName);

/* A continuacion se itera sobre el buffer almacenando cada objeto
 * encontrado, representado por un salto de linea ("\n") */
std::ostream_iterator<float> bufferIterator (bufferFichero, "\n");

/* Se copia la data del Iterador (bufferIterator) en el vector */
std::copy (dataVector.begin(), dataVector.end(), bufferIterator);
}

/* La siguiente funcion representa la conversion de una matriz Eigen a fichero */
void Extraer::EigenToFile(Eigen::MatrixXd dataMatriz, std::string fileName){
/* Se crea un buffer (bus de memoria temporal) como objeto que
 * contiene la data de un fichero */
std::ofstream bufferFichero(fileName);

/* Se condiciona mientraz el fichero este abierto almacenar
 * los datos, separados por salto de linea "\n". */
if(bufferFichero.is_open()){
    bufferFichero << dataMatriz << "\n";
}
}

/* Para determinar que tan bueno es nuestro modelo, vamos a crear una funcion
* como metrica de rendimiento. La metrica seleccionada es R2_score. */
float Extraer::R2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
auto numerador = pow((y - y_hat).array(), 2).sum();
auto denominador = pow((y.array() - y.mean()).array(), 2).sum();

return (1 - (numerador/denominador));
}
