// quadeval: method-agnostic quality report for any quad mesh against its source.
#include <cstdio>
#include <fstream>
#include <string>

#include "quadremesh/quadremesh.h"

using namespace qrm;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf(
            "usage: quadeval INPUT_TRI OUTPUT_QUAD.obj [-o report.json] [--target N] [--feature-angle DEG]\n"
            "                [--min-feature-length K] [--method NAME] [--runtime SECONDS] [--samples N]\n");
        return 1;
    }
    std::string in = argv[1], quad = argv[2], out, method;
    EvalParams ep;
    double runtime = -1;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-o") out = next();
        else if (a == "--target") ep.target_faces = std::stoi(next());
        else if (a == "--feature-angle") ep.feature_angle = std::stod(next());
        else if (a == "--min-feature-length") ep.min_feature_length = std::stod(next());
        else if (a == "--method") method = next();
        else if (a == "--runtime") runtime = std::stod(next());
        else if (a == "--samples") ep.samples = std::stoi(next());
        else if (a == "--coverage-tol") ep.coverage_tol = std::stod(next());
        else if (a == "--coverage-angle") ep.coverage_angle = std::stod(next());
    }
    try {
        TriMesh m = load_mesh(in);
        QuadMesh q = load_quad_obj(quad);
        std::string js = evaluate(m, q, ep, runtime, method);
        if (out.empty()) std::fputs(js.c_str(), stdout);
        else std::ofstream(out) << js;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
