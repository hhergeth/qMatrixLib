#pragma once

#include "qMatrix.h"
#include "qLinearSolver.h"
#include "qEigenSolver.h"

template<typename T, int M, int N> CUDA_FUNC_IN int svd(const qMatrix<T, M, N>& A, qMatrix<T, M, M>& U, qMatrix<T, N, N>& V, qMatrix<T, M, N>& S, T eps = T(-1))
{
	qMatrix<T, M, N> B = A;
	std::cout << "A : " << std::endl << A << std::endl;
	U = qMatrix<T, M, M>::Id();
	V = qMatrix<T, N, N>::Id();
	for (int k = 0; k < DMIN2(M - 1, N); k++)
	{
		qMatrix<T, M, 1> u = householder(B, k);
		qMatrix<T, M, M> U_k = qMatrix<T, M, M>::Id() - T(2) * u * u.Transpose();
		B = U_k.Transpose() * B;

		qMatrix<T, N, N> V_k = qMatrix<T, N, N>::Id();
		if (k < N - 2)
		{
			qMatrix<T, 1, N> b = B.row(k), v;
			for (int i = 0; i <= k; i++)
				b(0, i) = 0;
			T alpha = qMatrix_sgn(b(0, k + 1)) * b.p_norm(T(2));
			v = b - alpha * qMatrix<T, M, 1>::e(k + 1).Transpose();
			v = v / v.p_norm(T(2));
			V_k = qMatrix<T, N, N>::Id() - T(2) * v.Transpose() * v;
		}

		U = U_k * U;
		V = V * V_k;
		B = B * V_k;
	}

	qMatrix<T, M, M> ev0, ev1, U2, V2;
	qrAlgorithmSymmetric(B * B.Transpose(), ev0, U2);
	qrAlgorithmSymmetric(B.Transpose() * B, ev1, V2);

	S = diagmat(ev0.diag().sqrt());
	U = U.Transpose() * U2;
	V = V * V2;

	std::cout << "U : " << std::endl << U << std::endl;
	std::cout << "S : " << std::endl << S << std::endl;
	std::cout << "V : " << std::endl << V << std::endl;
	std::cout << "USV' : " << std::endl << (U * S * V.Transpose()) << std::endl;
	return 1;
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