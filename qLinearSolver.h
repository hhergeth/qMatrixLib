#pragma once

#include "qMatrix.h"
#include <assert.h>

//decompositions

namespace __qrHousholder__
{
	template<typename T, int M, typename S1> CUDA_FUNC_IN qMatrix<T, M, 1> householder(const qMatrix<T, M, 1, S1>& a)
	{
		qMatrix<T, M, 1> e = ::e<qMatrix<T, M, 1>>(0), u, v;
		T alpha = qMatrix_sgn(a(0, 0)) * a.p_norm(T(2));
		u = a - alpha * e;
		v = u / u.p_norm(T(2));
		return v;
	}

	template<bool UPPER = true, typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, 1> householderCol(const qMatrix<T, M, N, S1>& A, int k, int off = 0)
	{
		qMatrix<T, M, 1> a = Clone(A.col(k)), e = ::e<qMatrix<T, M, 1>>(k + off), u, v;
		if (a.accu() == a(0))
			return a / a(0);
		if (UPPER)
			for (int i = 0; i < k + off; i++)
				a(i) = 0;
		else for (int i = k + off + 1; i < M; i++)
				a(i) = 0;
		T alpha = qMatrix_sgn(a(k + off)) * a.p_norm(T(2));
		u = a - alpha * e;
		if (u.p_norm(T(2)) == 0)
			return qMatrix<T, M, 1>::Zero();
		v = u / u.p_norm(T(2));
		return v;
	}

	template<bool UPPER = true, typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, 1, N> householderRow(const qMatrix<T, M, N, S1>& A, int k, int off = 0)
	{
		qMatrix<T, 1, N> a = Clone(A.row(k)), e = ::e<qMatrix<T, 1, N>>(k + off), u, v;
		if (a.accu() == a(0))
			return a / a(0);
		if (UPPER)
			for (int i = 0; i < k + off; i++)
				a(i) = 0;
		else for (int i = k + off + 1; i < N; i++)
				a(i) = 0;
		T alpha = qMatrix_sgn(a(k + off)) * a.p_norm(T(2));
		u = a - alpha * e;
		if (u.p_norm(T(2)) == 0)
			return qMatrix<T, 1, N>::Zero();
		v = u / u.p_norm(T(2));
		return v;
	}
}

template<typename T, int M, int N, typename S1, typename S2, typename S3> CUDA_FUNC_IN void qrHousholder(const qMatrix<T, M, N, S1>& A, qMatrix<T, M, M, S2>& Q, qMatrix<T, M, N, S3>& R)
{
	Q.id();
	R = A;
	int K = DMIN2(M - 1, N);
	for (int k = 0; k < K; k++)
	{
		qMatrix<T, M, 1> v = __qrHousholder__::householderCol(R, k);
		qMatrix<T, M, M> Q_k = qMatrix<T, M, M>::Id() - T(2.0) * v * v.transpose();
		Q = Q * Q_k.transpose();
		R = Q_k * R;
	}
}

template<typename T, int M, int N, typename S1, typename S2, typename S3> CUDA_FUNC_IN void qrGivens(const qMatrix<T, M, N, S1>& A, qMatrix<T, M, M, S2>& Q, qMatrix<T, M, N, S3>& R)
{
	Q.id();
	R = A;
	for (int j = 0; j < DMIN2(N - 1, M); j++)
		for (int i = j + 1; i < M; i++)
		{
			T a = R(j, j), b = R(i, j), r, c, s;
			if (std::abs(b) < T(1e-5))
				continue;
			if (b == 0)
			{
				c = (T)_copysign(T(1), a);
				s = 0;
				r = std::abs(a);
			}
			else if (a == 0)
			{
				c = 0;
				s = -(T)_copysign(T(1), b);
				r = std::abs(b);
			}
			else if (std::abs(b) > std::abs(a))
			{
				T t = a / b, u = (T)_copysign(std::sqrt(T(1) + t * t), b);
				s = -T(1) / u;
				c = -s * t;
				r = b * u;
			}
			else
			{
				T t = b / a, u = (T)_copysign(std::sqrt(T(1) + t * t), a);
				c = T(1) / u;
				s = -c * t;
				r = a * u;
			}

			qMatrix<T, M, M> Q_k = qMatrix<T, M, M>::Id();
			Q_k(i, i) = Q_k(j, j) = c;
			Q_k(j, i) = -s;
			Q_k(i, j) = s;
			Q = Q * Q_k.transpose();
			R = Q_k * R;
		}
}

template<typename T, int N, typename S1, typename S2, typename S3, typename S4> CUDA_FUNC_IN void luDecomposition(const qMatrix<T, N, N, S1>& A, qMatrix<T, N, N, S2>& P, qMatrix<T, N, N, S3>& L, qMatrix<T, N, N, S4>& U)
{
	qMatrix<T, N, N> LR = A;
	int p[N];
	for (int i = 0; i < N; i++)
		p[i] = i;
	for (int j = 0; j < N - 1; j++)
	{
		int i_p = j;
		for (int i = 0; i < N; i++)
			if (std::abs(LR(i, j)) > std::abs(LR(i_p, j)))
				i_p = j;
		qMatrix_swap(p[i_p], p[j]);
		LR.swap_rows(j, i_p);
		for (int i = j + 1; i < N; i++)
		{
			LR(i, j) = LR(i, j) / LR(j, j);
			for (int k = j + 1; k < N; k++)
				LR(i, k) = LR(i, k) - LR(i, j) * LR(j, k);
		}
	}
	L = U = LR;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			if (i == j)
				L(i, j) = 1;
			if (i < j)
				L(i, j) = 0;
			if (i > j)
				U(i, j) = 0;
		}
	P.zero();
	for (int i = 0; i < N; i++)
		P(i, p[i]) = 1;
}

template<typename VEC, typename S1> CUDA_FUNC_IN VEC solveUpperDiagonal(const qMatrix<typename VEC::ELEMENT_TYPE, VEC::SIZE::DIM, VEC::SIZE::DIM, S1>& U, const VEC& rhs)
{
	VEC r;
	for (int i = VEC::SIZE::DIM - 1; i >= 0; i--)
	{
		typename VEC::ELEMENT_TYPE val = 0;
		for (int j = i + 1; j < VEC::SIZE::DIM; j++)
			val += U(i, j) * r(j, 0);
		r(i) = (rhs(i, 0) - val) / U(i, i);
	}
	return r;
}

template<typename VEC, typename S1> CUDA_FUNC_IN VEC solveLowerDiagonal(const qMatrix<typename VEC::ELEMENT_TYPE, VEC::SIZE::DIM, VEC::SIZE::DIM, S1>& L, const VEC& rhs)
{
	VEC r;
	for (int i = 0; i < VEC::SIZE::DIM; i++)
	{
		typename VEC::ELEMENT_TYPE val = 0;
		for (int j = 0; j < i; j++)
			val += L(i, j) * r(j, 0);
		r(i) = (rhs(i, 0) - val) / L(i, i);
	}
	return r;
}

template<typename T, int N, typename S1, typename S2, typename S3, typename S4> CUDA_FUNC_IN qMatrix<T, N, 1> solve(const qMatrix<T, N, N, S1>& P, const qMatrix<T, N, N, S2>& L, const qMatrix<T, N, N, S3>& U, const qMatrix<T, N, 1, S4>& rhs)
{
	qMatrix<T, N, 1> b = P * rhs;
	qMatrix<T, N, 1> d = solveLowerDiagonal(L, b);
	return solveUpperDiagonal(U, d);
}

template<typename T, int N, typename S1, typename S2, typename S3> CUDA_FUNC_IN qMatrix<T, N, 1> solve(const qMatrix<T, N, N, S1>& Q, const qMatrix<T, N, N, S2>& R, const qMatrix<T, N, 1, S3>& rhs)
{
	qMatrix<T, N, 1> b = Q.Transpose() * rhs;
	return solveUpperDiagonal(R, b);
}

template<typename T, int N, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, N, 1> solve(const qMatrix<T, N, N, S1>& A, const qMatrix<T, N, 1, S2>& rhs)
{
	qMatrix<T, N, N> L, U, P;
	luDecomposition(A, P, L, U);
	return solve(P, L, U, rhs);
}

template<typename T, int N, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, N, 1> conjugate_gradient(const qMatrix<T, N, N, S1>& A, const qMatrix<T, N, 1, S2>& rhs, int n = 100)
{
	qMatrix<T, N, 1> x;
	x.zero();
	qMatrix<T, N, 1> r = rhs - A * x;
	qMatrix<T, N, 1> p = r;
	T rsold = r.transpose() * r;
	for (int i = 0; i < n; i++)
	{
		qMatrix<T, N, 1> Ap = A * p;
		T alpha = rsold / (p.transpose() * Ap);
		x = x + alpha * p;
		r = r - alpha * Ap;
		qMatrix<T, N, 1> rCorr = rhs - A * x;
		T rsnew = r.transpose() * r;
		if (std::sqrt(rsnew) < 1e-5)
			break;
		p = r + rsnew / rsold * p;
		rsold = rsnew;
	}
	return x;
}

template<typename T, int N, typename S1, typename S2> CUDA_FUNC_IN qMatrix<T, N, 1> tridiag_solve(const qMatrix<T, N, N, S1>& A, const qMatrix<T, N, 1, S2>& rhs)
{
	assert(is_tridiagonal(A));
	qMatrix<T, N, 1> c_d, d_d;
	c_d(0) = A(0, 1) / A(0, 0);
	for (int i = 1; i < N - 1; i++)
		c_d(i) = A(i, i + 1) / (A(i, i) - A(i, i - 1) * c_d(i - 1));
	d_d(0) = rhs(0) / A(0, 0);
	for (int i = 1; i < N; i++)
		d_d(i) = (rhs(i) - A(i, i - 1) * d_d(i - 1)) / (A(i, i) - A(i, i - 1) * c_d(i - 1));
	qMatrix<T, N, 1> r;
	r(N - 1) = d_d(N - 1);
	for (int i = N - 2; i >= 0; i--)
		r(i) = d_d(i) - c_d(i) * r(i + 1);
	return r;
}