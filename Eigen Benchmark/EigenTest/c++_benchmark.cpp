//Benchmarking Eigen 
//Author:Hariharan Ramshankar
#define EIGEN_USE_MKL_ALL //to test MKL improvement
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
//#include <Windows.h>
using namespace std;
using namespace Eigen;
int main()
{
	
	//Fill random matrix 
	int size = 500;
	MatrixXd m = MatrixXd::Random(size, size);
	//start counter
	//auto wcts = std::chrono::system_clock::now();
	auto t1 = std::chrono::system_clock::now();
	//Do stuff
	//m.inverse();
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(m);
	if (eigensolver.info() != Success) abort();
	auto t2 = std::chrono::system_clock::now();
	//auto fp_ms = (std::chrono::system_clock::now() - wcts);
	std::chrono::duration<double, std::milli> fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;
	std::getchar();
	return 0;
}