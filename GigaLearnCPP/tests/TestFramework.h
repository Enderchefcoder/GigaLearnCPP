#pragma once

// Minimal single-header test framework
// Register tests with TEST_CASE(name) { ... } and use the CHECK_*() macros inside

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace GGLTest {

	struct TestCase {
		std::string name;
		std::function<void()> fn;
	};

	// Test registry (function-local static so it works across translation units)
	inline std::vector<TestCase>& GetTests() {
		static std::vector<TestCase> tests = {};
		return tests;
	}

	struct TestRegistrar {
		TestRegistrar(const std::string& name, std::function<void()> fn) {
			GetTests().push_back({ name, fn });
		}
	};

	// Thrown when a CHECK fails
	struct TestFailure : public std::runtime_error {
		using std::runtime_error::runtime_error;
	};

	inline int RunAllTests() {
		auto& tests = GetTests();

		int numPassed = 0;
		std::vector<std::string> failures = {};

		std::cout << "Running " << tests.size() << " test(s)..." << std::endl;

		for (auto& test : tests) {
			std::cout << " > " << test.name << "... " << std::flush;
			try {
				test.fn();
				std::cout << "PASSED" << std::endl;
				numPassed++;
			} catch (std::exception& e) {
				std::cout << "FAILED" << std::endl;
				std::cout << "    " << e.what() << std::endl;
				failures.push_back(test.name);
			}
		}

		std::cout << std::string(40, '=') << std::endl;
		std::cout << "Passed " << numPassed << "/" << tests.size() << " test(s)" << std::endl;

		if (!failures.empty()) {
			std::cout << "FAILED test(s):" << std::endl;
			for (auto& name : failures)
				std::cout << " > " << name << std::endl;
			return 1;
		}

		return 0;
	}
}

#define TEST_CASE(name) \
	static void _TestFn_##name(); \
	static GGLTest::TestRegistrar _testRegistrar_##name(#name, _TestFn_##name); \
	static void _TestFn_##name()

#define _TEST_FAIL(msgStream) { \
	std::stringstream _stream; \
	_stream << "At " << __FILE__ << ":" << __LINE__ << ": " << msgStream; \
	throw GGLTest::TestFailure(_stream.str()); \
}

#define CHECK_TRUE(cond) { if (!(cond)) _TEST_FAIL("CHECK_TRUE(" #cond ") failed"); }
#define CHECK_FALSE(cond) { if (cond) _TEST_FAIL("CHECK_FALSE(" #cond ") failed"); }

#define CHECK_EQ(a, b) { \
	auto _a = (a); auto _b = (b); \
	if (!(_a == _b)) _TEST_FAIL("CHECK_EQ(" #a ", " #b ") failed: " << _a << " != " << _b); \
}

#define CHECK_NEAR(a, b, tolerance) { \
	double _a = (double)(a); double _b = (double)(b); double _tol = (double)(tolerance); \
	if (std::isnan(_a) || std::isnan(_b) || std::abs(_a - _b) > _tol) \
		_TEST_FAIL("CHECK_NEAR(" #a ", " #b ") failed: " << _a << " != " << _b << " (tolerance: " << _tol << ")"); \
}

#define CHECK_THROWS(expr) { \
	bool _threw = false; \
	try { expr; } catch (...) { _threw = true; } \
	if (!_threw) _TEST_FAIL("CHECK_THROWS(" #expr ") failed: no exception thrown"); \
}
