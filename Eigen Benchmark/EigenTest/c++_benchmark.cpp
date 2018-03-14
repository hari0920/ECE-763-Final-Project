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
	cout << nbThreads() << endl; //Displays the Number of Parallel Threads Eigen will use
	cout<<"Eigenvalue"<<endl;
	//Start Timer
	auto t1 = std::chrono::system_clock::now();
	//Fill random matrix 
	int size = 1000;
	MatrixXd m = MatrixXd::Random(size, size);
	//Do stuff
	//m.inverse();
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(m);
	if (eigensolver.info() != Success) abort();
	auto t2 = std::chrono::system_clock::now();
	//auto fp_ms = (std::chrono::system_clock::now() - wcts);
	std::chrono::duration<double, std::milli> fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;

	cout<<"SVD"<<endl;
	//Start Timer
	t1 = std::chrono::system_clock::now();
	//Fill random matrix
	//size = 1000;
	m = MatrixXd::Random(size, size);
	//Do stuff
	//comparing SVD approaches
	if (size > 10000)
	{
		Eigen::BDCSVD<MatrixXd> svdsolver(m);
		svdsolver.compute(m);
	}
	else
	{
		Eigen::JacobiSVD<MatrixXd> jacobisvd(m);
		jacobisvd.compute(m);
	}
	t2 = std::chrono::system_clock::now();
	//auto fp_ms = (std::chrono::system_clock::now() - wcts);
	fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;

	cout<<"Inverse"<<endl;
	//Start Timer
	t1 = std::chrono::system_clock::now();
	//Fill random matrix
	//size = 1000;
	m = MatrixXd::Random(size, size);
	//Do stuff
	m.inverse();
	t2 = std::chrono::system_clock::now();
	//auto fp_ms = (std::chrono::system_clock::now() - wcts);
	fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;

	cout<<"Determinant"<<endl;
	//Start Timer
	t1 = std::chrono::system_clock::now();
	//Fill random matrix
	//size = 1000;
	m = MatrixXd::Random(size, size);
	//Do stuff
	m.determinant();
	t2 = std::chrono::system_clock::now();
	//auto fp_ms = (std::chrono::system_clock::now() - wcts);
	fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;

	cout<<"Dot"<<endl;
	//Start Timer
	t1 = std::chrono::system_clock::now();
	//Fill random matrix
	//size = 1000;
	m = MatrixXd::Random(size, size);
	//Do stuff
	MatrixXd b = m.inverse();
	MatrixXd result = m * b;
	t2 = std::chrono::system_clock::now();
	//auto fp_ms = (std::chrono::system_clock::now() - wcts);
	fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;

	std::cout << "All Tests Done"<< std::endl;

	std::getchar();
	return 0;
}
