#pragma once

#include "qMatrix.h"
#include "qLinearSolver.h"
#include "qEigenSolver.h"

#include <limits>

namespace __svd__
{
	template<typename T, int M, int N, typename S1> CUDA_FUNC_IN void givensL(qMatrix<T, M, N, S1>& S, int m, const T& a, const T& b)
	{
		T r = sqrt(a*a + b*b);
		T c = a / r;
		T s = -b / r;

		for (int i = 0; i < N; i++)
		{
			T s0 = S(m + 0, i);
			T s1 = S(m + 1, i);
			S(m, i) += s0*(c - 1);
			S(m, i) += s1*(-s);

			S(m + 1, i) += s0*(s);
			S(m + 1, i) += s1*(c - 1);
		}
	}

	template<typename T, int M, int N, typename S1> CUDA_FUNC_IN void givensR(qMatrix<T, M, N, S1>& S, int m, const T& a, const T& b)
	{
		T r = sqrt(a*a + b*b);
		T c = a / r;
		T s = -b / r;

		for (int i = 0; i < N; i++)
		{
			T s0 = S(i, m + 0);
			T s1 = S(i, m + 1);
			S(i, m) += s0*(c - 1);
			S(i, m) += s1*(-s);

			S(i, m + 1) += s0*(s);
			S(i, m + 1) += s1*(c - 1);
		}
	}
}

template<typename T, int M, int N, typename S1, typename S2, typename S3, typename S4> CUDA_FUNC_IN void svd(const qMatrix<T, M, N, S1>& A, qMatrix<T, M, M, S2>& U, qMatrix<T, N, N, S3>& V, qMatrix<T, M, N, S4>& S, T eps = T(-1))
{
	qMatrix<T, M, N> B = A;
	U = qMatrix<T, M, M>::Id();
	V = qMatrix<T, N, N>::Id();
	for (int k = 0; k < N; k++)
	{
		qMatrix<T, M, 1> u = __qrHousholder__::householderCol(B, k);
		if (u.is_zero())
			break;
		qMatrix<T, M, M> Q_k = qMatrix<T, M, M>::Id() - T(2) * u * u.transpose();

		B = Q_k * B;
		U = U * Q_k;

		if (k < N - 2)
		{
			qMatrix<T, 1, N> v = __qrHousholder__::householderRow(B, k, 1);
			if (v.is_zero())
				break;
			qMatrix<T, N, N> P_k1 = qMatrix<T, N, N>::Id() - T(2) * v.transpose() * v;

			B = B * P_k1;
			V = P_k1 * V;
		}
	}
	//LOG_MAT(B);
	//LOG_MAT(U*B*V.transpose());
	//LOG_MAT(A);
	qMatrix<T, M, 1> evU;
	qMatrix<T, M, M> U2;
	qrAlgorithmSymmetric(B * B.transpose(), evU, U2);
	qMatrix<T, N, 1> evV;
	qMatrix<T, N, N> V2;
	qrAlgorithmSymmetric(B.transpose() * B, evV, V2);
	S = diagmat<T, M, N>(evU.sqrt());
	U = U * U2;
	V = V * V2;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, M> pseudoinverse(const qMatrix<T, M, N, S1>& A)
{
	qMatrix<T, M, M> U;
	qMatrix<T, N, N> V;
	qMatrix<T, M, N> E;
	svd(A, U, V, E);
	for (int i = 0; i < DMIN2(M, N); i++)
		if (E(i, i) != 0)
			E(i, i) = T(1.0) / E(i, i);
	return V * E * U.transpose();
}

//general purpose
template<typename T, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, N> inv(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> L, U, P, I;
	luDecomposition(A, P, L, U);
	for (int i = 0; i < N; i++)
		I.col(i) = solve(P, L, U, e<qMatrix<T, N, 1>>(i));
	return I;
}

template<typename T, int N, typename S1> CUDA_FUNC_IN T det(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> L, U, P;
	luDecomposition(A, P, L, U);
	T det = 1;
	for (int i = 0; i < N; i++)
		det *= L(i, i) * U(i, i);
	return det;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN qMatrix<T, M, N> null(const qMatrix<T, N, M, S1>& A, int& rank, const T& eps = T(1e-5) * DMAX2(M, N))
{
	qMatrix<T, M, M> U;
	qMatrix<T, N, N> V;
	qMatrix<T, M, N> E;
	svd(A, U, V, E);
	rank = 0;
	qMatrix<T, M, N> nul = qMatrix<T, M, N>::Zero();
	for (int i = 0; i < DMIN2(M, N); i++)
	{
		if (E(i, i) < eps)
		{
			nul.col(rank) = V.col(i);
			rank++;
		}
	}
	return nul;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN int rank(const qMatrix<T, M, N, S1>& A)
{
	int r;
	null(A, r);
	return r;
}

template<typename T, int N, typename S1, typename S2, typename S3> CUDA_FUNC_IN void eig(const qMatrix<T, N, N, S1>& A, qMatrix<T, N, N, S2>& values, qMatrix<T, N, N, S3>& vectors)
{
	qrAlgorithm(A, values, vectors);
	values = diagmat(values.diag());
}

template<typename T, int N, typename S1> CUDA_FUNC_IN qMatrix<T, N, 1> eig(const qMatrix<T, N, N, S1>& A)
{
	qMatrix<T, N, N> values, vectors;
	eig(A, values, vectors);
	return values.diag();
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN T cond(const qMatrix<T, M, N, S1>& A)
{
	qMatrix<T, M, M> U;
	qMatrix<T, N, N> V;
	qMatrix<T, M, N> S;
	int n = svd(A, U, V, S);
	T la = 0, sm = 0;
	for (int i = 0; i < n; i++)
	{
		la = std::max(la, S(i, i));
		sm = std::min(sm, S(i, i));
	}
	return la / sm;
}

template<typename T, int N, typename S1> CUDA_FUNC_IN T cond(const qMatrix<T, N, N, S1>& A)
{
	return A.p_norm(T(2)) * inv(A).p_norm(T(2));
}