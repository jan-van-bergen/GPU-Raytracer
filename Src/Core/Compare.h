#pragma once

namespace Compare {
	template<typename T>
	struct Equal {
		bool operator()(const T & a, const T & b) const {
			return a == b;
		}
	};

	template<typename T>
	struct NotEqual {
		bool operator()(const T & a, const T & b) const {
			return a != b;
		}
	};

	template<typename T>
	struct LessThan {
		bool operator()(const T & a, const T & b) const {
			return a < b;
		}
	};

	template<typename T>
	struct LessThanEqual {
		bool operator()(const T & a, const T & b) const {
			return a <= b;
		}
	};

	template<typename T>
	struct GreaterThan {
		bool operator()(const T & a, const T & b) const {
			return a > b;
		}
	};

	template<typename T>
	struct GreaterThanEqual {
		bool operator()(const T & a, const T & b) const {
			return a >= b;
		}
	};
}
