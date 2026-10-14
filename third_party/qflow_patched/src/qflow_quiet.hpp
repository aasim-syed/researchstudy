// quadremesh patch: route QuadriFlow's console output through a verbosity flag and turn its
// internal `exit()` calls into exceptions so a failing component can be retried instead of
// terminating the host process (CLI or Python interpreter).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
namespace qflow { extern int g_verbose; }
using qflow::g_verbose;
#define printf(...) (qflow::g_verbose ? std::printf(__VA_ARGS__) : 0)
#define exit(code) throw std::runtime_error(std::string("quadriflow internal failure at ") + __FILE__ + ":" + std::to_string(__LINE__))
