#pragma once

#include "qMatrix.h"
#include "qLinearSolver.h"
#include "qEigenSolver.h"

#include <limits>

template<typename T, int M, int N, typename S1, typename S2, typename S3, typename S4> CUDA_FUNC_IN void bidiagonalisation(const qMatrix<T, M, N, S1>& A, qMatrix<T, M, M, S2>& U, qMatrix<T, N, N, S4>& V, qMatrix<T, M, N, S3>& B)
{
	static_assert(M >= N, "Only valid for M >= N matrices A");
	B = A;
	U = qMatrix<T, M, M>::Id();
	V = qMatrix<T, N, N>::Id();
	for (int k = 0; k < N; k++)
	{
		qMatrix<T, M, 1> u = __qrHousholder__::householderCol(B, k);
		if (!u.is_zero())
		{
			qMatrix<T, M, M> Q_k = qMatrix<T, M, M>::Id() - T(2) * u * u.transpose() / float(u.transpose() * u);

			B = Q_k * B;
			U = U * Q_k;
		}
		if (k < N - 2)
		{
			qMatrix<T, 1, N> v = __qrHousholder__::householderRow(B, k, 1);
			if (!v.is_zero())
			{
				qMatrix<T, N, N> P_k1 = qMatrix<T, N, N>::Id() - T(2) * v.transpose() * v / float(v * v.transpose());

				B = B * P_k1;
				V = V * P_k1;
			}
		}
	}
}

template<typename T, int M, int N, typename S1, typename S2, typename S3, typename S4> CUDA_FUNC_IN int svd(const qMatrix<T, M, N, S1>& A, qMatrix<T, M, M, S2>& U, qMatrix<T, N, N, S3>& V, qMatrix<T, M, N, S4>& S, int n = 50)
{
	qMatrix<T, M, N> B;
	bidiagonalisation(A, U, V, B);

	qMatrix<T, M, 1> evU;
	qMatrix<T, M, M> U2;
	int nEigenvalues1 = qrAlgorithm(B * B.transpose(), evU, U2, n);
	qMatrix<T, N, 1> evV;
	qMatrix<T, N, N> V2;
	int nEigenvalues2 = qrAlgorithm(B.transpose() * B, evV, V2, n);

	if (nEigenvalues1 != nEigenvalues2)
		return -1;

	S = diagmat<T, M, N>(evU.sqrt());
	LOG_MAT(U); LOG_MAT(V); LOG_MAT(V2);
	for (int i = nEigenvalues1; i < M; i++)
		U2.col(i) = e<qMatrix<float, M, 1>>(i);
	for (int i = nEigenvalues2; i < N; i++)
		V2.col(i) = e<qMatrix<float, N, 1>>(i);

	//fix singular vector directions
	for (int i = 0; i < M; i++)
		if (U2(0, i) < 0)
			U2.col(i) = -U2.col(i);
	for (int i = 0; i < N; i++)
		if (V2(0, i) < 0)
			V2.col(i) = -V2.col(i);

	U = U * U2;
	V = V * V2;

	return nEigenvalues1;
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

//computes left/right null space and returns the rank of the matrix
template<typename T, int M, int N, typename S1, typename S2, typename S3> CUDA_FUNC_IN int null(const qMatrix<T, M, N, S1>& A, qMatrix<T, M, M, S2>& leftNull, qMatrix<T, N, N, S3>& rightNull, const T& eps = T(1e-5))
{
	qMatrix<T, M, M> U;
	qMatrix<T, N, N> V;
	qMatrix<T, M, N> E;
	int rank = svd(A, U, V, E);
	LOG_MAT(U); LOG_MAT(E); LOG_MAT(V);
	if (rank == -1)
		return -1;
	leftNull.zero();
	rightNull.zero();
	for (int i = rank; i < N; i++)
	{
		leftNull.row(i - rank) = (U.transpose()).row(i);
		rightNull.col(i - rank) = V.col(i);
	}
	return rank;
}

template<typename T, int M, int N, typename S1> CUDA_FUNC_IN int rank(const qMatrix<T, M, N, S1>& A)
{
	qMatrix<T, M, M> U;
	qMatrix<T, N, N> V;
	qMatrix<T, M, N> E;
	return svd(A, U, V, E);//the rank of a matrix corresponds to the number of nonzero singular values
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