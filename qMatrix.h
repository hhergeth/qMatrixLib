#pragma once

#include <iostream>     // std::cout
#include <sstream>      // std::istringstream
#include <string>       // std::string
#include <iomanip>
#include <algorithm>
#include <cmath>

#ifndef CUDA_FUNC_IN
#define CUDA_FUNC_IN inline
#endif

#define MATRIX_ELEMENT_FUNC(FUNC_HEADER, FUNC) \
	CUDA_FUNC_IN qMatrix<T, M, N> FUNC_HEADER const \
		{ \
		qMatrix<T, M, N> res; \
		for(int i = 0; i < M; i++) \
			for(int j = 0; j < N; j++) \
				res(i, j) = FUNC; \
		return res; \
		}

#define MATRIX_ELEMENT_FUNC_2(NAME, FUNC) \
	MATRIX_ELEMENT_FUNC(NAME(), FUNC(operator()(i, j)))

#define MATRIX_ELEMENT_FUNC_3(NAME) \
	MATRIX_ELEMENT_FUNC_2(NAME, std::NAME)

#ifndef DMIN2
#define DMIN2(A, B) ((A) < (B) ? (A) : (B))
#endif

#ifndef DMAX2
#define DMAX2(A, B) ((A) > (B) ? (A) : (B))
#endif

template <typename T> CUDA_FUNC_IN int qMatrix_sgn(T val)
{
	return (T(0) < val) - (val < T(0));
}

template <typename T> CUDA_FUNC_IN T qMatrix_round(T val)
{
	if (val < 0) return (T)std::ceil(val - 0.5);
	return (T)std::floor(val + 0.5);
}

template<typename T> CUDA_FUNC_IN void qMatrix_swap(T& a, T& b)
{
	T tmp = a;
	a = b;
	b = tmp;
}

template<typename T> CUDA_FUNC_IN T qMatrix_sqr(T val)
{
	return val * val;
}

struct MatrixDataStorageBase
{
	
};

template<typename T, int M, int N> struct MatrixDataStorage_Value : public MatrixDataStorageBase
{
private:
	T dat[DMAX2(M * N, 1)];
public:
	CUDA_FUNC_IN const T& operator()(int i, int j) const
	{
		if (i >= M || j >= N)
#ifndef ISCUDA
			throw std::runtime_error("Invalid matrix access.");
#else
			printf("%s   Invalid matrix element access at (%d, %d)!\n", __PRETTY_FUNCTION__, i, j);
#endif
		return dat[i * N + j];
	}
	CUDA_FUNC_IN T& operator()(int i, int j)
	{
		if (i >= M || j >= N)
#ifndef ISCUDA
			throw std::runtime_error("Invalid matrix access.");
#else
			printf("%s   Invalid matrix element access at (%d, %d)!\n", __PRETTY_FUNCTION__, i, j);
#endif
		return dat[i * N + j];
	}
};

template<typename MAT, typename T> struct MatrixDataStorage_Ref : public MatrixDataStorageBase
{
private:
	MAT& ref_mat;
	int off_i, off_j;
public:
	CUDA_FUNC_IN MatrixDataStorage_Ref(MAT& ref, int off_i, int off_j)
		: ref_mat(ref), off_i(off_i), off_j(off_j)
	{

	}
	CUDA_FUNC_IN const T& operator()(int i, int j) const
	{
		return (T)ref_mat(off_i + i, off_j + j);
	}
	CUDA_FUNC_IN T& operator()(int i, int j)
	{
		return ref_mat(off_i + i, off_j + j);
	}
};

template<typename MAT, typename T> struct MatrixDataStorage_Ref_Const : public MatrixDataStorageBase
{
private:
	const MAT& ref_mat;
	int off_i, off_j;
public:
	CUDA_FUNC_IN MatrixDataStorage_Ref_Const(const MAT& ref, int off_i, int off_j)
		: ref_mat(ref), off_i(off_i), off_j(off_j)
	{

	}
	CUDA_FUNC_IN const T& operator()(int i, int j) const
	{
		return (T)ref_mat(off_i + i, off_j + j);
	}
};

template<typename T, int M, int N, typename STORAGE = MatrixDataStorage_Value<T, M, N>> struct qMatrix
{
	static_assert(std::is_base_of<MatrixDataStorageBase, STORAGE>::value, "Storage type must be derived from MatrixDataStorageBase!");
	STORAGE m_storage;
public:

	enum SIZE
	{
		ROWS = M,
		COLS = N,
		DIM = DMAX2(M, N),
	};

	typedef T ELEMENT_TYPE;
	typedef STORAGE STORAGE_TYPE;
	typedef qMatrix<T, M, N> MATRIX_TYPE_VAL;

	typedef qMatrix<T, M, 1> COL_TYPE;
	typedef qMatrix<T, 1, N> ROW_TYPE;

	CUDA_FUNC_IN qMatrix<T, M, N, STORAGE>()
	{

	}

	CUDA_FUNC_IN qMatrix<T, M, N, STORAGE>(const STORAGE& storage)
		: m_storage(storage)
	{

	}

	CUDA_FUNC_IN const T& operator()(int i, int j) const
	{
		return m_storage(i, j);
	}

	CUDA_FUNC_IN T& operator()(int i, int j)
	{
		//in case there is an compiler error here(C2440) "cannot convert from const float to float&"
		//the problem is trying to set element of an const matrix
		return m_storage(i, j);
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Zero()
	{
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r(i, j) = 0;
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Id()
	{
		qMatrix<T, M, N> r;
		r.id();
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Ones()
	{
		qMatrix<T, M, N> r;
		r.ones();
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Parse(const char* s)
	{
		std::istringstream iss(s);
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
			{
				T f;
				iss >> f;
				r(i, j) = f;
			}
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Vandermonde(const qMatrix<T, 1, M>& x)
	{
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				r(i, j) = std::pow(x(0, i), j);
			}
		}
		return r;
	}

	CUDA_FUNC_IN void id()
	{
		zero();
		for (int i = 0; i < DMIN2(M, N); i++)
			operator()(i, i) = 1;
	}

	CUDA_FUNC_IN void zero()
	{
		fill(T(0));
	}

	CUDA_FUNC_IN void ones()
	{
		fill(T(1));
	}

	CUDA_FUNC_IN void fill(const T& val)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				operator()(i, j) = val;
	}

	CUDA_FUNC_IN const T& operator()(int i) const
	{
		if (is_colvec())
			return operator()(i, 0);
		else if (is_rowvec())
			return operator()(0, i);
#ifndef __CUDA_ARCH__
		else throw std::runtime_error("Invalid matrix access.");
#else
		else
		{
			printf("%s   Matrix is not a vector!\n", __PRETTY_FUNCTION__);
			return operator()(0, 0);
		}
#endif 
	}

	CUDA_FUNC_IN T& operator()(int i)
	{
		if (N == 1)//col
			return operator()(i, 0);
		else if (M == 1)//row
			return operator()(0, i);
#ifndef __CUDA_ARCH__
		else throw std::runtime_error("Invalid matrix access.");
#else
		else
		{
			printf("%s   Matrix is not a vector!\n", __PRETTY_FUNCTION__);
			return operator()(0, 0);
		}
#endif 
	}

	CUDA_FUNC_IN bool operator==(const qMatrix<T, M, N>& rhs) const
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				if (operator()(i, j) != rhs(i, j))
					return false;
		return true;
	}

	CUDA_FUNC_IN bool operator!=(const qMatrix<T, M, N>& rhs) const
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				if (operator()(i, j) != rhs(i, j))
					return true;
		return false;
	}

	CUDA_FUNC_IN size_t n_rows() const
	{
		return M;
	}

	CUDA_FUNC_IN size_t n_cols() const
	{
		return N;
	}

private:
	typedef MatrixDataStorage_Ref<qMatrix<T, M, N, STORAGE>, T> REF_STORAGE;
	typedef MatrixDataStorage_Ref_Const<qMatrix<T, M, N, STORAGE>, T> REF_CONST_STORAGE;
public:
	//first_row, first_col, last_row, last_col
	template<int p, int r, int q, int s> CUDA_FUNC_IN qMatrix<T, q - p + 1, s - r + 1, REF_STORAGE> submat()
	{
		return qMatrix<T, q - p + 1, s - r + 1, REF_STORAGE>(REF_STORAGE(*this, p, r));
	}

	template<int p, int r, int q, int s> CUDA_FUNC_IN qMatrix<T, q - p + 1, s - r + 1, REF_CONST_STORAGE> submat() const
	{
		return qMatrix<T, q - p + 1, s - r + 1, REF_CONST_STORAGE>(REF_CONST_STORAGE(*this, p, r));
	}

	//selects columns ie p < M, q < M, p <= q
	template<int p, int q> CUDA_FUNC_IN qMatrix<T, N, q - p + 1, REF_STORAGE> cols()
	{
		return submat<0, p, M, q>();
	}

	template<int p, int q> CUDA_FUNC_IN qMatrix<T, N, q - p + 1, REF_CONST_STORAGE> cols() const
	{
		return submat<0, p, M, q>();
	}

	//selects rows ie r < N, s < N, r <= s
	template<int r, int s> CUDA_FUNC_IN qMatrix<T, s - r + 1, M, REF_STORAGE> rows()
	{
		return submat<r, 0, s, N>();
	}

	template<int r, int s> CUDA_FUNC_IN qMatrix<T, s - r + 1, M, REF_CONST_STORAGE> rows() const
	{
		return submat<r, 0, s, N>();
	}

	CUDA_FUNC_IN qMatrix<T, 1, N, REF_STORAGE> row(int i)
	{
		return qMatrix<T, 1, N, REF_STORAGE>(REF_STORAGE(*this, i, 0));
	}

	CUDA_FUNC_IN qMatrix<T, 1, N, REF_CONST_STORAGE> row(int i) const
	{
		return qMatrix<T, 1, N, REF_CONST_STORAGE>(REF_CONST_STORAGE(*this, i, 0));
	}

	CUDA_FUNC_IN qMatrix<T, M, 1, REF_STORAGE> col(int j)
	{
		return qMatrix<T, M, 1, REF_STORAGE>(REF_STORAGE(*this, 0, j));
	}

	CUDA_FUNC_IN qMatrix<T, M, 1, REF_CONST_STORAGE> col(int j) const
	{
		return qMatrix<T, M, 1, REF_CONST_STORAGE>(REF_CONST_STORAGE(*this, 0, j));
	}

	CUDA_FUNC_IN void swap_rows(int r, int s)
	{
		for (int j = 0; j < N; j++)
			qMatrix_swap(operator()(r, j), operator()(s, j));
	}

	CUDA_FUNC_IN void swap_cols(int p, int q)
	{
		for (int i = 0; i < M; i++)
			qMatrix_swap(operator()(i, p), operator()(i, q));
	}

	CUDA_FUNC_IN qMatrix<T, M, N> fliplr() const
	{
		qMatrix<T, M, N> res;
		for (int j = 0; j < N; j++)
			res.col(N - j - 1) = col(j);
		return res;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> flipud() const
	{
		qMatrix<T, M, N> res;
		for (int i = 0; i < M; i++)
			res.row(M - i - 1) = row(i);
		return res;
	}

	CUDA_FUNC_IN qMatrix<T, N, M> transpose() const
	{
		qMatrix<T, N, M> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r(j, i) = operator()(i, j);
		return r;
	}

	template<int R, typename S2> CUDA_FUNC_IN qMatrix<T, M, N + R> JoinHorizontal(const qMatrix<T, M, R, S2>& rhs) const
	{
		qMatrix<T, M, N + R> res;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N + R; j++)
				res(i, j) = j < N ? operator()(i, j) : rhs(i, j - N);
		return res;
	}

	template<int R, typename S2> CUDA_FUNC_IN qMatrix<T, M + R, N> JoinVertical(const qMatrix<T, R, N, S2>& rhs) const
	{
		qMatrix<T, M + R, N> res;
		for (int i = 0; i < M + R; i++)
			for (int j = 0; j < N; j++)
				res(i, j) = i < M ? operator()(i, j) : rhs(i - M, j);
		return res;
	}

	template<typename S2> CUDA_FUNC_IN qMatrix<T, M, N> MulElement(const qMatrix<T, M, N, S2>& rhs) const
	{
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) * rhs(i, j);
		return r;
	}

	template<typename S2> CUDA_FUNC_IN qMatrix<T, M, N> DivElement(const qMatrix<T, M, N, S2>& rhs) const
	{
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) / rhs(i, j);
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> operator++() const
	{
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) + 1;
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> operator--() const
	{
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) - 1;
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> operator-() const
	{
		qMatrix<T, M, N> r;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r(i, j) = -operator()(i, j);
		return r;
	}

	MATRIX_ELEMENT_FUNC(clamp(const T& low, const T& high), operator()(i, j) < low ? low : (operator()(i, j) > high ? high : operator()(i, j)))

	MATRIX_ELEMENT_FUNC_3(abs)

	MATRIX_ELEMENT_FUNC_3(exp)

	MATRIX_ELEMENT_FUNC_3(log)

	MATRIX_ELEMENT_FUNC_3(sqrt)

	MATRIX_ELEMENT_FUNC(pow(const T& p), std::pow(operator()(i, j), p))

	MATRIX_ELEMENT_FUNC(sqr(), operator()(i, j) * operator()(i, j))

	MATRIX_ELEMENT_FUNC_3(floor)

	MATRIX_ELEMENT_FUNC_3(ceil)

	MATRIX_ELEMENT_FUNC_2(sign, signf)

	MATRIX_ELEMENT_FUNC_3(cos)
	MATRIX_ELEMENT_FUNC_3(acos)
	MATRIX_ELEMENT_FUNC_3(cosh)
	//MATRIX_ELEMENT_FUNC_3(acosh)

	MATRIX_ELEMENT_FUNC_3(sin)
	MATRIX_ELEMENT_FUNC_3(asin)
	MATRIX_ELEMENT_FUNC_3(sinh)
	//MATRIX_ELEMENT_FUNC_3(asinh)

	MATRIX_ELEMENT_FUNC_3(tan)
	MATRIX_ELEMENT_FUNC_3(atan)
	MATRIX_ELEMENT_FUNC_3(tanh)
	//MATRIX_ELEMENT_FUNC_3(atanh)

	//accumulates all elements
	CUDA_FUNC_IN T accu() const
	{
		T res = T();
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				res += operator()(i, j);
		return res;
	}

	CUDA_FUNC_IN T accuabs() const
	{
		T res = T();
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				res += std::abs(operator()(i, j));
		return res;
	}

	//accumulates all elements in the specified sub matrix, inclusive bounds
	CUDA_FUNC_IN T accu(int row_start, int col_start, int row_end, int col_end) const
	{
		T res = T();
		for (int i = row_start; i <= row_end; i++)
			for (int j = col_start; j <= col_end; j++)
				res += std::abs(operator()(i, j));
		return res;
	}

	//accumulates the absolute of all elements in the specified sub matrix, inclusive bounds
	CUDA_FUNC_IN T accuabs(int row_start, int col_start, int row_end, int col_end) const
	{
		T res = T();
		for (int i = row_start; i <= row_end; i++)
			for (int j = col_start; j <= col_end; j++)
				res += std::abs(operator()(i, j));
		return res;
	}

	//the minimal entry
	CUDA_FUNC_IN T min() const
	{
		T res = std::numeric_limits<T>::max();
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				res = std::min(res, operator()(i, j));
		return res;
	}

	//the maximal entry
	CUDA_FUNC_IN T max() const
	{
		T res = std::numeric_limits<T>::min();
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				res = std::max(res, operator()(i, j));
		return res;
	}

	//the mean of all entries
	CUDA_FUNC_IN T mean() const
	{
		return accu() / T(M * N);
	}

	//the variance of all entries
	CUDA_FUNC_IN T var() const
	{
		T res = T(), m = mean();
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
			{
				T f = operator()(i, j) - m;
				res += f * f;
			}
		return res / T(M * N);
	}

	CUDA_FUNC_IN T stddev() const
	{
		return std::sqrt(var());
	}

	CUDA_FUNC_IN T p_norm(T p) const
	{
		T r = T();
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				r += ::pow(operator()(i, j), p);
		return ::pow(r, T(1.0) / p);
	}

	CUDA_FUNC_IN T col_sum_norm() const
	{
		T r = T();
		for (int j = 0; j < N; j++)
		{
			T s = T(0);
			for (int i = 0; i < M; i++)
				s += std::abs(operator()(i, j));
			r = std::max(r, s);
		}
		return r;
	}

	CUDA_FUNC_IN T row_sum_norm() const
	{
		T r = T();
		for (int i = 0; i < M; i++)
		{
			T s = T(0);
			for (int j = 0; j < N; j++)
				s += std::abs(operator()(i, j));
			r = std::max(r, s);
		}
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Round(int digts_after_decimal) const
	{
		qMatrix<T, M, N> res;
		T f = std::pow((T)10, (T)(digts_after_decimal));
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				res(i, j) = qMatrix_round(operator()(i, j) * f) / f;
		return res;
	}

	//test if no element is NAN or inf
	CUDA_FUNC_IN bool is_finite() const
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				if (operator()(i, j) != operator()(i, j) || !std::isfinite(operator()(i, j)))
					return false;
		return true;
	}

	CUDA_FUNC_IN bool is_vec() const
	{
		return is_colvec() || is_rowvec();
	}

	CUDA_FUNC_IN bool is_colvec() const
	{
		return N == 1;
	}

	CUDA_FUNC_IN bool is_rowvec() const
	{
		return M == 1;
	}

	CUDA_FUNC_IN bool is_quadratic() const
	{
		return M == N;
	}

	CUDA_FUNC_IN bool is_symmetric() const
	{
		if (!is_quadratic())
			return false;
		for (int i = 0; i < M; i++)
			for (int j = i + 1; j < N; j++)
				if (operator()(i, j) != operator()(j, i))
					return false;
		return true;
	}

	CUDA_FUNC_IN bool is_upper_triangular() const
	{
		for (int i = 1; i < M; i++)
			for (int j = 0; j < i; j++)
				if (operator()(i, j) != 0)
					return false;
		return true;
	}

	CUDA_FUNC_IN bool is_lower_triangular() const
	{
		for (int j = 1; j < N; j++)
			for (int i = 0; i < j; i++)
				if (operator()(i, j) != 0)
					return false;
		return true;
	}

	CUDA_FUNC_IN bool is_upper_bidiagonal() const
	{
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < i; j++)
				if (operator()(i, j) != 0)
					return false;

			for (int j = i + 2; j < N; j++)
				if (operator()(i, j) != 0)
					return false;
		}
		return true;
	}

	CUDA_FUNC_IN bool is_lower_bidiagonal() const
	{
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < i - 1; j++)
				if (operator()(i, j) != 0)
					return false;

			for (int j = i + 1; j < N; j++)
				if (operator()(i, j) != 0)
					return false;
		}
		return true;
	}

	CUDA_FUNC_IN bool is_bidiagonal() const
	{
		return is_upper_bidiagonal() || is_lower_bidiagonal();
	}

	CUDA_FUNC_IN bool is_zero() const
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				if (operator()(i, j) != 0)
					return false;
		return true;
	}

	CUDA_FUNC_IN bool is_orthogonal(const T& eps = T(1e-5f)) const
	{
		T l = (*this * this->transpose() - qMatrix<T, M, N>::Id()).p_norm(T(2));
		return l < eps;
	}

	CUDA_FUNC_IN int num_negative_elements() const
	{
		int n = 0;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				n += operator()(i, j) < 0;
		return n;
	}

	template<typename U> CUDA_FUNC_IN qMatrix<U, M, N> convert()
	{
		qMatrix<U, M, N> res;
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				res(i, j) = U(operator()(i, j));
		return res;
	}

	template<typename U> CUDA_FUNC_IN operator qMatrix<U, M, N>()
	{
		return convert<U>();
	}

	CUDA_FUNC_IN operator T() const
	{
		static_assert(M == 1 && N == 1, "Matrix not of size 1x1!");
		return operator()(0, 0);
	}

	void print(std::ostream &os) const
	{
		std::ostringstream str;
		size_t w = 0;
		for (int i = 0; i < M; ++i)
			for (int j = 0; j < N; ++j)
			{
				str << operator()(i, j);
				w = DMAX2((size_t)str.tellp(), w);
				str.str("");
			}

		for (int i = 0; i < M; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				str.str("");
				str << operator()(i, j);
				size_t l = (size_t)str.tellp();
				os << std::right << std::string(w + 1 - l, ' ') << operator()(i, j);
			}

			os << std::endl;
		}
	}

	std::string ToString() const
	{
		std::stringstream str;
		Round(4).print(str);
		return str.str();
	}

	std::string ToString(const std::string var_name) const
	{
		return var_name + " = \n" + Round(4).ToString();
	}

	friend std::ostream & operator<<(std::ostream &os, const qMatrix<T, M, N, STORAGE>& p)
	{
		p.Round(3).print(os);
		return os;
	}

	template<typename S2> CUDA_FUNC_IN qMatrix<T, M, N, STORAGE>& operator= (const qMatrix<T, M, N, S2>& rhs)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				operator()(i, j) = rhs(i, j);
		return *this;
	}

	CUDA_FUNC_IN qMatrix<T, M, N, STORAGE>& operator= (const qMatrix<T, M, N, STORAGE>& rhs)
	{
		if(this != &rhs)
		{
			for (int i = 0; i < M; i++)
				for (int j = 0; j < N; j++)
					operator()(i, j) = rhs(i, j);
		}
		return *this;
	}
};

//does a copy of the matrix to a new object, helper for ref matrices
template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> Clone(qMatrix<T, M, N, S1> const& lhs)
{
	qMatrix<T, M, N> res;
	res = lhs;
	return res;
}

template<typename T, int M, int N, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, M, N> operator+(qMatrix<T, M, N, S1> const& lhs, qMatrix<T, M, N, S2> const& rhs)
{
	qMatrix<T, M, N> r;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			r(i, j) = lhs(i, j) + rhs(i, j);
	return r;
}

template<typename T, int M, int N, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, M, N> operator-(qMatrix<T, M, N, S1> const& lhs, qMatrix<T, M, N, S2> const& rhs)
{
	qMatrix<T, M, N> r;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			r(i, j) = lhs(i, j) - rhs(i, j);
	return r;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> operator+(qMatrix<T, M, N, S1> const& lhs, T const& rhs)
{
	qMatrix<T, M, N> r;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			r(i, j) = lhs(i, j) + rhs;
	return r;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> operator-(qMatrix<T, M, N, S1> const& lhs, T const& rhs)
{
	qMatrix<T, M, N> r;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			r(i, j) = lhs(i, j) - rhs;
	return r;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> operator*(qMatrix<T, M, N, S1> const& lhs, const T& rhs)
{
	qMatrix<T, M, N> r;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			r(i, j) = lhs(i, j) * rhs;
	return r;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> operator/(qMatrix<T, M, N, S1> const& lhs, const T& rhs)
{
	return lhs * (T(1.0) / rhs);
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> operator*(const T& lhs, const qMatrix<T, M, N, S1>& rhs)
{
	return rhs * lhs;
}

template<typename T, int M, int N, int R, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, M, R> operator*(qMatrix<T, M, N, S1> const& lhs, qMatrix<T, N, R, S2> const& rhs)
{
	qMatrix<T, M, R> r;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < R; j++)
		{
			T val = 0;
			for (int k = 0; k < N; k++)
				val += lhs(i, k) * rhs(k, j);
			r(i, j) = val;
		}
	return r;
}

template<typename VEC, int M, int N, typename S1>  CUDA_FUNC_IN VEC diag(const qMatrix<typename VEC::ELEMENT_TYPE, M, N, S1>& A)
{
	VEC res;
	for (int i = 0; i < VEC::SIZE::DIM; i++)
		res(i, 0) = A(i, i);
	return res;
}

template<typename VEC, int M, int N, typename S1> CUDA_FUNC_IN void diag(qMatrix<typename VEC::ELEMENT_TYPE, M, N, S1>& A, const VEC& d)
{
	for (int i = 0; i < VEC::SIZE::DIM; i++)
		A(i, i) = d(i);
}

template<typename VEC> CUDA_FUNC_IN qMatrix<typename VEC::ELEMENT_TYPE, VEC::SIZE::DIM, VEC::SIZE::DIM> diagmat(const VEC& diag)
{
	qMatrix<typename VEC::ELEMENT_TYPE, VEC::SIZE::DIM, VEC::SIZE::DIM> res;
	res.zero();
	::diag(res, diag);
	return res;
}

template<typename T, int M, int N, int L, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> diagmat(const qMatrix<T, L, 1, S1>& diag)
{
	qMatrix<T, M, N> res;
	res.zero();
	for (int i = 0; i < DMIN2(DMIN2(M, N), L); i++)
		res(i, i) = diag(i);
	return res;
}

template<typename VEC> CUDA_FUNC_IN static VEC e(int i)
{
	VEC r = VEC::Zero();
	r(i) = 1.0f;
	return r;
}

template<typename VEC> CUDA_FUNC_IN VEC linspace(const typename VEC::ELEMENT_TYPE& start, const typename VEC::ELEMENT_TYPE& end, const typename VEC::ELEMENT_TYPE& n)
{
	typename VEC::ELEMENT_TYPE f = (end - start) / (n - 1);
	VEC res;
	for (int i = 0; i < VEC::DIM; i++)
		res(i) = start + f * i;
	return res;
}

template<typename VEC> CUDA_FUNC_IN VEC linspace(const typename VEC::ELEMENT_TYPE& start, const typename VEC::ELEMENT_TYPE& end)
{
	return linspace(start, end, 100);
}

template<typename VEC> CUDA_FUNC_IN typename VEC::ELEMENT_TYPE norm(const VEC& v, const typename VEC::ELEMENT_TYPE p = typename VEC::ELEMENT_TYPE(2))
{
	return v.p_norm(p);
}

namespace __kronecker_product__
{
	template<typename T, int M, int N, int P, int R, int i, int j, typename S1, typename S2, typename S3> struct loop
	{
		CUDA_FUNC_IN static void exec(const qMatrix<T, M, N, S1>& lhs, const qMatrix<T, P, R, S2>& rhs, qMatrix<T, M * P, N * R, S3>& res)
		{
			res.template submat<P * i, R * j, P * (i + 1) - 1, R * (j + 1) - 1>() = lhs(i, j) * rhs;
			loop<T, M, N, P, R, i + 1, j, S1, S2, S3>::exec(lhs, rhs, res);
		}
	};

	template<typename T, int M, int N, int P, int R, int j, typename S1, typename S2, typename S3> struct loop<T, M, N, P, R, M, j, S1, S2, S3>
	{
		CUDA_FUNC_IN static void exec(const qMatrix<T, M, N, S1>& lhs, const qMatrix<T, P, R, S2>& rhs, qMatrix<T, M * P, N * R, S3>& res)
		{

		}
	};

	template<typename T, int M, int N, int P, int R, int COL, typename S1, typename S2, typename S3> struct loopStarter
	{
		CUDA_FUNC_IN static void exec(const qMatrix<T, M, N, S1>& lhs, const qMatrix<T, P, R, S2>& rhs, qMatrix<T, M * P, N * R, S3>& res)
		{
			loop<T, M, N, P, R, 0, COL, S1, S2, S3>::exec(lhs, rhs, res);
			loopStarter<T, M, N, P, R, COL + 1, S1, S2, S3>::exec(lhs, rhs, res);
		}
	};

	template<typename T, int M, int N, int P, int R, typename S1, typename S2, typename S3> struct loopStarter<T, M, N, P, R, N, S1, S2, S3>
	{
		CUDA_FUNC_IN static void exec(const qMatrix<T, M, N, S1>& lhs, const qMatrix<T, P, R, S2>& rhs, qMatrix<T, M * P, N * R, S3>& res)
		{
		}
	};
}
template<typename T, int M, int N, int P, int R, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, M * P, N * R> kronecker_product(const qMatrix<T, M, N, S1>& lhs, const qMatrix<T, P, R, S2>& rhs)
{
	qMatrix<T, M * P, N * R> res;
	res.id();
	__kronecker_product__::loopStarter<T, M, N, P, R, 0, S1, S2, typename decltype(res)::STORAGE_TYPE>::exec(lhs, rhs, res);
	return res;
}

template<typename T, int N, typename S1> CUDA_FUNC_IN bool is_tridiagonal(const qMatrix<T, N, N, S1>& A)
{
	return is_upper_hessenberg(A) && is_lower_hessenberg(A);
}

template<typename T, int N, typename S1> CUDA_FUNC_IN bool is_upper_hessenberg(const qMatrix<T, N, N, S1>& A)
{
	for (int i = 2; i < N; i++)
		for (int j = 0; j < i - 1; j++)
			if (A(i, j) != 0)
				return false;
	return true;
}

template<typename T, int N, typename S1> CUDA_FUNC_IN bool is_lower_hessenberg(const qMatrix<T, N, N, S1>& A)
{
	for (int j = 2; j < N; j++)
		for (int i = 0; i < j - 1; i++)
			if (A(i, j) != 0)
				return false;
	return true;
}

//interpret A as a symmetric mat, copying upper to lower
template<typename T, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, N> symmatu(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> res = A;
	for (int i = 1; i < N; i++)
		for (int j = 0; j < i; j++)
			res(i, j) = A(j, i);
	return res;
}

//interpret A as a symmetric mat, copying lower to upper
template<typename T, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, N> symmatl(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> res = A;
	for (int j = 1; j < N; j++)
		for (int i = 0; i < j; i++)
			res(i, j) = A(j, i);
	return res;
}

//interpret square matrix A as upper triangular
template<typename T, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, N> trimatu(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> res = A;
	for (int i = 1; i < N; i++)
		for (int j = 0; j < i; j++)
			res(i, j) = T(0);
	return res;
}

//interpret square matrix A as lower triangular
template<typename T, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, N> trimatl(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> res = A;
	for (int j = 1; j < N; j++)
		for (int i = 0; i < j; i++)
			res(i, j) = T(0);
	return res;
}

//interpret square matrix A as tridiagonal
template<typename T, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, N> tridiag(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> res = A;
	for (int j = 1; j < N; j++)
		for (int i = 0; i < j - 1; i++)
			res(i, j) = T(0);
	for (int i = 1; i < N; i++)
		for (int j = 0; j < i - 1; j++)
			res(i, j) = T(0);
	return res;
}


template<typename T, int N, typename S1> CUDA_FUNC_IN T trace(const qMatrix<T, N, N, S1>& A)
{
	return A.diag().accu();
}

template<typename T, int M, int N, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, M, N> minimize(const qMatrix<T, M, N, S1>& A, const qMatrix<T, M, N, S2>& B)
{
	qMatrix<T, M, N> res;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			res(i, j) = std::min(A(i, j), B(i, j));
	return res;
}

template<typename T, int M, int N, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, M, N> maximize(const qMatrix<T, M, N, S1>& A, const qMatrix<T, M, N, S2>& B)
{
	qMatrix<T, M, N> res;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			res(i, j) = std::max(A(i, j), B(i, j));
	return res;
}
