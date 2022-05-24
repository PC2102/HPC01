#ifndef REGRESIONLINEAL_H
#define REGRESIONLINEAL_H

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>

class RegresionLineal
{
public:
    RegresionLineal(){}

    float fCostoOLS(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);

    std::tuple <Eigen::VectorXd, std::vector<float>> GradientDes(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha,int iteraciones);
};

#endif // REGRESIONLINEAL_H
