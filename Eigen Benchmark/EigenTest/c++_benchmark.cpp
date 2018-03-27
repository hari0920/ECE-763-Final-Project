//Benchmarking Eigen 
//Author:Hariharan Ramshankar
//#define EIGEN_USE_MKL_ALL //to test MKL improvement (uncomment in Windows Visual Studio)
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
//#include <Windows.h>
using namespace std;
using namespace Eigen;
int main()
{
	cout << nbThreads() << endl; //Displays the Number of Parallel Threads Eigen will use
	
	int i = 10;
	for(int size=i;size<500;size*=5)
	{
	cout<<endl;
	cout<<"Size:"<<size<<endl;
	cout << "Eigenvalue" << endl;
	//Start Timer
	auto t1 = std::chrono::system_clock::now();
	//Fill random matrix 
	MatrixXd m = MatrixXd::Random(size, size);
	//Do stuff
	//m.inverse();
	m.eigenvalues();
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
	if (size > 5000)
	{
		m.bdcSvd();
	}
	else
	{
		m.jacobiSvd();
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
	//size = 10000;
	m = MatrixXd::Random(size, size);
	//Do stuff
	MatrixXd result = m*m.inverse();
	t2 = std::chrono::system_clock::now();
	//auto fp_ms = (std::chrono::system_clock::now() - wcts);
	fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;
	

	cout<<"Testing Multi-Core Performance"<<endl;
	//Start Timer
	t1 = std::chrono::system_clock::now();
	int n = 10000;
	MatrixXd A = MatrixXd::Ones(n, n);
	MatrixXd B = MatrixXd::Ones(n, n);
	MatrixXd C = MatrixXd::Ones(n, n);
	C.noalias() += A * B;
	std::cout << C.sum() << "\n";
	t2 = std::chrono::system_clock::now();
	fp_ms1 = t2 - t1;
	std::cout << "Finished in " << fp_ms1.count() << "ms" << std::endl;
	}
	std::cout << "All Tests Done" << std::endl;
	//std::getchar();
return 0;
}
