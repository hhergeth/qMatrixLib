#pragma once

#include "qMatrix.h"

template<typename T, int M, int N, typename S1> std::string ToWolframString(const qMatrix<T, M, N, S1>& A)
{
	std::ostringstream stream;
	stream << "{";
	for (int row = 0; row < M; row++)
	{
		stream << "{";
		for (int col = 0; col < N; col++)
		{
			stream << A(row, col);
			if (col != N - 1)
				stream << ", ";
		}
		stream << "}" << (row != M - 1 ? "," : "") << "\n";
	}
	stream << "}";
	return stream.str();
}

template<typename T, int M, int N> qMatrix<T, M, N> FromWolframString(const std::string& str)
{
	std::istringstream stream(str);
	qMatrix<T, M, N> r;
	r.zero();
	std::string::size_type p = str.find("{");
	std::string::size_type n_row = str.find("{", p + 1);
	T val;
	for (int row = 0; row < M; row++)
	{
		p = n_row;
		if (p == std::string::npos)
			break;
		n_row = str.find("{", p + 1);
		for (int col = 0; col < N; col++)
		{
			std::string::size_type p2 = str.find(",", p + 1);
			stream.seekg(p + 1);
			stream >> val;
			r(row, col) = val;
			if (p2 > n_row)
				break;
			p = p2;
		}
	}
	return r;
}

template<typename T, int M, int N, typename S1> std::string ToMatlabString(const qMatrix<T, M, N, S1>& A)
{
	std::ostringstream stream;
	stream << "[";
	for (int row = 0; row < M; row++)
	{
		for (int col = 0; col < N; col++)
		{
			stream << A(row, col);
			if (col != N - 1)
				stream << ", ";
		}
		if (row != M - 1)
			stream << ";\n";
	}
	stream << "]";
	return stream.str();
}

template<typename T, int M, int N> qMatrix<T, M, N> FromMatlabString(const std::string& str)
{
	std::istringstream stream(str);
	qMatrix<T, M, N> r;
	r.zero();
	std::string::size_type p = str.find("[");
	T val;
	for (int row = 0; row < M; row++)
	{
		std::string::size_type n_row = str.find(";", p + 1);
		for (int col = 0; col < N; col++)
		{
			stream.seekg(p + 1);
			stream >> val;
			r(row, col) = val;
			std::string::size_type p2 = str.find(",", p + 1);
			if (p2 > n_row)
				break;
			p = p2;
		}
		p = n_row;
	}
	return r;
}

template<typename T, int M, int N> struct MAT
{
	int i;
	qMatrix<T, M, N> m;
	CUDA_FUNC_IN MAT()
		: i(0)
	{
		m.zero();
	}
	CUDA_FUNC_IN MAT& operator%(const T& val)
	{
		m(i / N, i % N) = val;
		i++;
		return *this;
	}
	CUDA_FUNC_IN MAT& operator()()
	{
		i += M - (i % M);
		return *this;
	}
	CUDA_FUNC_IN operator qMatrix<T, M, N>() const
	{
		return m;
	}
};

template<typename T, int N> struct VEC : public MAT<T, N, 1>
{
};

#ifndef int64
typedef long long int64;
#endif

#define MAKE_TYPEDEF(L, T) \
	typedef MAT<T, L, 1> q##T##L;  \
	typedef MAT<T, L, L> q##T##L##x##L; 

#define MAKE_ALL_TYPEDEFS(L) \
	MAKE_TYPEDEF(L, int) \
	MAKE_TYPEDEF(L, int64) \
	MAKE_TYPEDEF(L, float) \
	MAKE_TYPEDEF(L, double)

MAKE_ALL_TYPEDEFS(1)
MAKE_ALL_TYPEDEFS(2)
MAKE_ALL_TYPEDEFS(3)
MAKE_ALL_TYPEDEFS(4)
MAKE_ALL_TYPEDEFS(5)
