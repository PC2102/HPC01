/*********************************************************
* Fecha: 23-05-2022
* Autor: Pedro Cardenas
* Materia: HPC-1
* Tema: Parcial 3 HPC
* ******************************************************/
// EL interfaz representa el menu de funciones disponible en las clases/biblioteca

#include "Extraccion/extraer.h"
#include "regresionlineal.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string.h>

int main(int argc, char *argv[]){
    /* Se crea un objeto del tipo Extraer
     * para incluir los 3 argumentos que necesita
     * el objeto. */
    Extraer extraerData(argv[1], argv[2], argv[3]);

    /* Se crea un objeto del tipo LinearRegression sin ningun argumento de entrada */
    RegresionLineal LR;

    /* Se requiere probar la lectura del fichero y
     * luego se requiere observar el dataset como
     * un objeto de matriz tipo dataframe. */
    std::vector<std::vector<std::string>> dataSET = extraerData.ReadCSV();
    int filas = dataSET.size()+1;
    int columnas = dataSET[0].size();
    Eigen::MatrixXd MatrizDataF = extraerData.CSVToEigen(
                dataSET, filas, columnas);

    /* Se imprime la matriz que tiene los datos del
     * dataset. */
    std::cout << "                      **** - Se imprime el Dataset - ****                     " << std::endl;
    std::cout << MatrizDataF << std::endl;
    std::cout << "Filas : " << filas << std::endl;
    std::cout << "Columnas: " << columnas << std::endl << std::endl << std::endl;

    /* Se imprime el promedio, Se debe validar */

    std::cout << "Promedio: " << extraerData.Promedio(MatrizDataF) << std::endl;
    std::cout << "Desviacion: " << extraerData.DesvStandard(MatrizDataF) << std::endl << std::endl << std::endl;;

    /* Se crea la matrix para almacenar la normalizacion */
    Eigen::MatrixXd matNormal = extraerData.Normalizador(MatrizDataF);
    /*std::cout << matNormal << std::endl;*/

    /* A continuacion se dividen "Entrenamiento" y "Prueba" el conjunto de
     * datos de entrada (matNormal). */
    Eigen::MatrixXd X_train, y_train, X_test, y_test;

    /* Se dividen los datos y el 80% es para "Entrenamiento". */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> MatrizDividida = extraerData.TrainTestSplit(matNormal, 0.8);

    /* Se desempaqueta la tupla. */
    std::tie(X_train, y_train, X_test, y_test) = MatrizDividida;

    /* std::cout << matNormal.rows() << std::endl;
    std::cout << X_test.cols() << std::endl;
    std::cout << y_test.cols() << std::endl;
    std::cout << X_train.rows() << std::endl;
    std::cout << y_train.rows() << std::endl;*/

    /* A continuacion se realizara el primer modulo de ML. Se hara
     * una clase "RegresionLineal". Con su correspondiente constructor
     * de argumentos de entrada y metodos para el calculo del modelo
     * RL(RegresionLineal). Se tiene en cuenta que el RL es un metodo
     * estadistico que define la relacion entre la variables dependiente
     * y las variables independientes.
     * La idea principal, es definir una linea recta(Hiperplano) con sus
     * coeficientes(pendientes) y punto de corte.
     * Se tienen diferentes metodos para resolver RL, para este caso se
     * usara el metodo de los Minimos Cuadrados Ordinarios(OLS), por ser
     * un metodo sencillo y computacionalmente economico.
     * Representa una solucion optima para un conjunto de datos no
     * complejos. El dataset a utilizar es el de "Vino rojo" el cual tiene
     * 11 variables (multivariable) independientes. Para ello hemos de
     * implementar el algoritmo del gradiente descendiente cuyo objetivo
     * principal es minimizar la funcion de costo. */

    /* Se define un vector para entrenamiento y para prueba inicializados en
     * unos (1) */
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /* Se redimensionan las matrices para ser ubicadas en el vectur de Unos:
     * similar a un numpy reshape */
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /* Se define el vector teta que pasara al algoritmo del gradiente
     * descendiente, basicamente es un vector de ceros (0) del mismo
     * tamaño del entrenamiento, adicional se pasara alpha y el numero
     * de iteraciones */

    /* Theta: Coeficientes , alpha = ratio de aprendizaje */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    /* A continuacion se definen las variables de salida que representan los
     * coeficientes y el vector de costo */
    Eigen::MatrixXd thetaSalida;
    std::vector<float> costo;

    /* Se desempaqueta la tupla como objeto instanciado del gradiente descendiente */
    /* Pendiente y punto de corte (thetaSalida y costo) */
    std::tuple<Eigen::VectorXd, std::vector<float>> objetoGradiente = LR.GradientDes(X_train, y_train, theta, alpha, iteraciones);
    std::tie(thetaSalida, costo) = objetoGradiente;

    /* Se imprimen los coheficientes para cada variable */
    //std::cout << thetaSalida << std::endl;

    /* Se imprime para inspeccion ocular la funcion de costo */
    /*for(auto v: costo){
        std::cout << v << std::endl;
    }*/

    /* Se almacena la funcion de costo y las variables theta a ficheros */
    extraerData.FiletoVector(costo, "costo.txt");
    extraerData.EigenToFile(thetaSalida, "theta.txt");

    /* Se calcula el promedio y la desviacion estandar, para calcular las predicciones
     * es decir se debe de normalizar para calcular la metrica */
    auto muData = extraerData.Promedio(MatrizDataF);
    auto muFeatures = muData(0, 4);
    auto escalado = MatrizDataF.rowwise() - MatrizDataF.colwise().mean();
    auto sigmaData = extraerData.DesvStandard(escalado);
    auto sigmaFeatures = sigmaData(0, 4);
    Eigen::MatrixXd y_train_hat = (X_train*thetaSalida*sigmaFeatures).array() + muFeatures;
    Eigen::MatrixXd y_test_hat = (X_test*thetaSalida*sigmaFeatures).array() + muFeatures;
    Eigen::MatrixXd y = MatrizDataF.col(4).topRows(41);
    float R2_score = extraerData.R2_score(y, y_train_hat);
    std::cout << R2_score << std::endl;

    extraerData.EigenToFile(y_train_hat, "y_train_hat.cpp");
    extraerData.EigenToFile(y_test_hat, "y_test_hat.cpp");


    return EXIT_SUCCESS;
}
