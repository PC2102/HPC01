#ifndef EXTRAER_H
#define EXTRAER_H
/*********************************************************
* Fecha: 23-05-2022
* Autor: Pedro Cardenas
* Materia: HPC-1
* Tema: Parcial 3 HPC
* ******************************************************/

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>

// .csv Valores separados por comas
// Fila es un registro
// Columna es los datos en el registro

class Extraer{
    /* Se presenta el constructor de los argumentos
     * de entrada a la clase "Extraer" */

    /* Nombre del dataset */
    std::string setDatos;

    /* Separador de columnas */
    std::string delimitador;

    /* Si tiene cabecera o no, el dataset*/
    bool header;

public:
    Extraer(std::string datos,
            std::string separador,
            bool head):
        setDatos(datos),
        delimitador(separador),
        header(head){}
    std::vector<std::vector<std::string>> ReadCSV();
    Eigen::MatrixXd CSVToEigen(
            std::vector<std::vector<std::string>>  SETdatos,
            int filas, int columnas);
    auto Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean());

    /* Funcion para calcular la Desviacion Estandar */
    /* Para implementar la desviacion estandar
     * datos = xi - x.promedio */
    auto DesvStandard(Eigen::MatrixXd datos) -> decltype((datos.array().square().colwise().sum() / (datos.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit( Eigen::MatrixXd DatosNorm, float sizeTrain);
    void FiletoVector(std::vector<float> dataVector, std::string fileName);
    void EigenToFile(Eigen::MatrixXd dataMatriz, std::string fileName);
    float R2_score(Eigen::MatrixXd y, Eigen::MatrixXd y_train_hat);


};
#endif // EXTRAER_H
