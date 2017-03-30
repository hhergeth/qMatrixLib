#pragma once

namespace __template_unroll__
{

template<int N> struct templateIntWrapper
{
	enum
	{
		VAL = N,
	};
};

template<int k, int N> struct forwardUnroll
{
	template<typename F> CUDA_FUNC_IN static void exec_iteration(F clb)
	{
		clb(templateIntWrapper<k>());
		forwardUnroll<k + 1, N>::exec_iteration(clb);
	}
};

template<int N> struct forwardUnroll<N, N>
{
	template<typename F> CUDA_FUNC_IN static void exec_iteration(F clb)
	{

	}
};

}

//unrolls i = k; i < N; i++
template<int k, int N, typename F> CUDA_FUNC_IN void for_i(F clb)
{
	__template_unroll__::forwardUnroll<k, N>::exec_iteration(clb);
}

//unrolls i = N; i >= k; i--
template<int N, int k, typename F> CUDA_FUNC_IN void for_i_down(F clb)
{
	__template_unroll__::forwardUnroll<k, N + 1>::exec_iteration([&clb](auto iT)
	{
		clb(__template_unroll__::templateIntWrapper<N - decltype(iT)::VAL>());
	});
}