#pragma once

#include "qMatrix.h"
#include "qLinearSolver.h"
#include "qEigenSolver.h"

template<typename T, int M, int N> CUDA_FUNC_IN void svd(const qMatrix<T, M, N>& A, qMatrix<T, M, M>& U, qMatrix<T, N, N>& V, qMatrix<T, M, N>& S, T eps = T(-1))
{
	qMatrix<T, M, N> B = A;
	U = qMatrix<T, M, M>::Id();
	V = qMatrix<T, N, N>::Id();
	for (int k = 0; k < N; k++)
	{
		qMatrix<T, M, 1> u = __qrHousholder__::householder(B, k);
		qMatrix<T, M, M> Q_k = qMatrix<T, M, M>::Id() - T(2) * u * u.transpose();

		B = Q_k * B;
		U = U * Q_k;

		if (k < N - 2)
		{
			qMatrix<T, N, 1> v = __qrHousholder__::householder(B.transpose(), k);
			qMatrix<T, N, N> P_k1 = qMatrix<T, N, N>::Id() - T(2) * v * v.transpose();

			B = B * P_k1;
			V = P_k1 * V;
		}
	}

	qMatrix<T, M, M> ev0, U2;
	qrAlgorithmSymmetric(B * B.transpose(), ev0, U2);
	qMatrix<T, N, N> V2, ev1;
	qrAlgorithmSymmetric(B.transpose() * B, ev1, V2);

	auto dq = diag<qMatrix<T, M, 1>>(ev0);
	S = diagmat<T, M, N>(dq.sqrt());
	U = U * U2;
	V = V2 * V;
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, N, M> pseudoinverse(const qMatrix<T, M, N>& A)
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
template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, N> inv(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> L, U, P, I;
	luDecomposition(A, P, L, U);
	for (int i = 0; i < N; i++)
		I.col(i, solve(P, L, U, e<qMatrix<T, N, 1>>(i)));
	return I;
}

template<typename T, int N> CUDA_FUNC_IN T det(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> L, U, P;
	luDecomposition(A, P, L, U);
	T det = 1;
	for (int i = 0; i < N; i++)
		det *= L(i, i) * U(i, i);
	return det;
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> null(const qMatrix<T, N, M>& A, int& rank, const T& eps = T(1e-5))
{
	qMatrix<T, M, N> R;
	qMatrix<T, M, M> Q;
	qrHousholder(A, Q, R);
	rank = 0;
	while (rank < DMIN2(M, N) && std::abs(R(rank, rank)) > eps)
		rank++;
	qMatrix<T, M, N> nul = qMatrix<T, M, N>::Zero();
	for (int i = 0; i < rank; i++)
		nul.col(i, Q.col(N - 1 - rank + i));
	return nul;
}

template<typename T, int M, int N> CUDA_FUNC_IN int rank(const qMatrix<T, M, N>& A)
{
	int r;
	null(A, r);
	return r;
}

template<typename T, int N> CUDA_FUNC_IN void eig(const qMatrix<T, N, N>& A, qMatrix<T, N, N>& values, qMatrix<T, N, N>& vectors)
{
	qrAlgorithm(A, values, vectors);
	values = diagmat(values.diag());
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> eig(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> values, vectors;
	eig(A, values, vectors);
	return values.diag();
}

template<typename T, int M, int N> CUDA_FUNC_IN T cond(const qMatrix<T, M, N>& A)
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

template<typename T, int N> CUDA_FUNC_IN T cond(const qMatrix<T, N, N>& A)
{
	return A.p_norm(T(2)) * inv(A).p_norm(T(2));
}