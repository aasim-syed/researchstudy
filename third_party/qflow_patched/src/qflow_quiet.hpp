// quadremesh patch: route QuadriFlow's console output through a verbosity flag.
#pragma once
#include <cstdio>
#include <iostream>
namespace qflow { extern int g_verbose; }
using qflow::g_verbose;
#define printf(...) (qflow::g_verbose ? std::printf(__VA_ARGS__) : 0)
