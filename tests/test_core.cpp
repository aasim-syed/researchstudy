// Minimal C++ regression tests: robustness on synthetic inputs, boundary preservation,
// symmetry, pure-quad mode and determinism.
#include <cmath>
#include <cstdio>
#include <string>

#include "quadremesh/geometry.h"
#include "quadremesh/quadremesh.h"

using namespace qrm;

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
            failures++;                                                          \
        }                                                                        \
    } while (0)

// open grid (plane with boundary), n x n vertices, triangulated
static TriMesh grid(int n, double bump = 0.0) {
    TriMesh m;
    m.V.resize(n * n, 3);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double x = i / double(n - 1), y = j / double(n - 1);
            m.V.row(i * n + j) << x, y, bump * std::sin(3 * x) * std::cos(2 * y);
        }
    m.F.resize(2 * (n - 1) * (n - 1), 3);
    int k = 0;
    for (int i = 0; i + 1 < n; ++i)
        for (int j = 0; j + 1 < n; ++j) {
            int a = i * n + j, b = (i + 1) * n + j, c = (i + 1) * n + j + 1, d = i * n + j + 1;
            m.F.row(k++) << a, b, c;
            m.F.row(k++) << a, c, d;
        }
    return m;
}

static double quad_ratio(const QuadMesh& q) {
    int n = 0;
    for (int i = 0; i < q.Q.rows(); ++i) n += q.Q(i, 3) >= 0;
    return q.Q.rows() ? double(n) / q.Q.rows() : 0.0;
}

int main(int argc, char** argv) {
    std::string models = argc > 1 ? argv[1] : "data/models";
    {
        std::printf("open surface with boundary\n");
        TriMesh g = grid(60, 0.1);
        Params p;
        p.target_faces = 400;
        Result r = quadremesh(g, p);
        CHECK(r.mesh.Q.rows() > 200 && r.mesh.Q.rows() < 800);
        CHECK(quad_ratio(r.mesh) > 0.98);
        // boundary vertices stay on the square boundary
        int nb = 0, bad = 0;
        for (int i = 0; i < r.mesh.V.rows(); ++i) {
            double x = r.mesh.V(i, 0), y = r.mesh.V(i, 1);
            double d = std::min(std::min(x, 1 - x), std::min(y, 1 - y));
            if (d < 1e-6) nb++;
            if (x < -1e-6 || x > 1 + 1e-6 || y < -1e-6 || y > 1 + 1e-6) bad++;
        }
        CHECK(nb > 40);
        CHECK(bad == 0);
    }
    {
        std::printf("determinism\n");
        TriMesh m = load_mesh(models + "/fandisk.obj");
        Params p;
        p.target_faces = 1500;
        Result a = quadremesh(m, p), b = quadremesh(m, p);
        CHECK(a.mesh.Q.rows() == b.mesh.Q.rows());
        CHECK(a.mesh.V.rows() == b.mesh.V.rows());
        if (a.mesh.V.rows() == b.mesh.V.rows()) CHECK((a.mesh.V - b.mesh.V).cwiseAbs().maxCoeff() == 0.0);
        if (a.mesh.Q.rows() == b.mesh.Q.rows()) CHECK((a.mesh.Q - b.mesh.Q).cwiseAbs().maxCoeff() == 0);
    }
    {
        std::printf("pure quad + symmetry\n");
        TriMesh m = load_mesh(models + "/spot.obj");
        Params p;
        p.target_faces = 2000;
        p.pure_quad = true;
        Result r = quadremesh(m, p);
        CHECK(quad_ratio(r.mesh) == 1.0);
        Params s;
        s.target_faces = 2000;
        s.symmetry_axis = 0;
        Result rs = quadremesh(m, s);
        CHECK(rs.mesh.Q.rows() > 1000);
    }
    {
        std::printf("non-manifold / noisy input\n");
        TriMesh m = load_mesh(models + "/fandisk_noisy.obj");
        Params p;
        p.target_faces = 2000;
        Result r = quadremesh(m, p);
        CHECK(quad_ratio(r.mesh) > 0.98);
        EvalParams ep;
        ep.target_faces = 2000;
        ep.samples = 20000;
        std::string js = evaluate(m, r.mesh, ep);
        CHECK(js.find("\"nonmanifold_edges\": 0") != std::string::npos);
    }
    std::printf(failures ? "%d FAILURES\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
