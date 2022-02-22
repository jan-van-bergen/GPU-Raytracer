#pragma once
#include <utility>

#include "OwnPtr.h"
#include "Allocators/Allocator.h"

template<typename> struct Function;

template<typename Result, typename ... Args>
struct Function<Result(Args ...)> {
	struct CallableBase {
		virtual ~CallableBase() = default;

		virtual Result call(Args ... args) const = 0;
	};

	template<typename T>
	struct Callable final : CallableBase {
		mutable T impl;

		Callable(T impl) : impl(std::move(impl)) { }

		DEFAULT_COPYABLE(Callable);
		DEFAULT_MOVEABLE(Callable);

		~Callable() = default;

		Result call(Args ... args) const override {
			return impl(std::forward<Args>(args) ...);
		}
	};

	OwnPtr<CallableBase> callable; // TODO: Perhaps support some in-situ storage here for small callables to avoid indirection

	Function() = default;

	template<typename T>
	Function(T && impl) : callable(make_owned<Callable<T>>(std::move(impl))) { }

	NON_COPYABLE(Function);
	DEFAULT_MOVEABLE(Function);

	~Function() = default;

	Result operator()(Args ... args) const {
		return callable->call(std::forward<Args>(args) ...);
	}

	operator bool() {
		return callable.ptr != nullptr;
	}
};
