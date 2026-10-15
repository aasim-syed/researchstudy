// pybind11 bindings for the quadremesh C++ core.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quadremesh/quadremesh.h"

namespace py = pybind11;
using namespace qrm;

namespace {

Params params_from_kwargs(const py::kwargs& kw) {
    Params p;
    for (auto item : kw) {
        std::string k = py::str(item.first);
        py::handle v = item.second;
        if (v.is_none()) continue;
        if (k == "target_faces") p.target_faces = v.cast<int>();
        else if (k == "target_edge_length") p.target_edge_length = v.cast<double>();
        else if (k == "feature_angle") p.feature_angle = v.cast<double>();
        else if (k == "min_feature_length") p.min_feature_length = v.cast<double>();
        else if (k == "preserve_boundary") p.preserve_boundary = v.cast<bool>();
        else if (k == "feature_refine_rounds") p.feature_refine_rounds = v.cast<int>();
        else if (k == "feature_loops") p.feature_loops = v.cast<bool>();
        else if (k == "feature_loop_budget") p.feature_loop_budget = v.cast<double>();
        else if (k == "match_face_count") p.match_face_count = v.cast<bool>();
        else if (k == "feature_loop_partial") p.feature_loop_partial = v.cast<bool>();
        else if (k == "curved_feature_splits") p.curved_feature_splits = v.cast<bool>();
        else if (k == "trace_all_features") p.trace_all_features = v.cast<bool>();
        else if (k == "triangle_budget") p.triangle_budget = v.cast<double>();
        else if (k == "singularity_budget") p.singularity_budget = v.cast<double>();
        else if (k == "feature_refine_max_len") p.feature_refine_max_len = v.cast<double>();
        else if (k == "feature_field_budget") p.feature_field_budget = v.cast<double>();
        else if (k == "coverage_mode") {
            if (v.cast<bool>()) {
                p.trace_all_features = p.feature_loops = p.feature_loop_partial = p.curved_feature_splits = true;
                p.feature_refine_rounds = 1; p.feature_refine_max_len = 8; p.singularity_budget = 25;
            }
        }
        else if (k == "curvature_weight") p.curvature_weight = v.cast<double>();
        else if (k == "curvature_min_anisotropy") p.curvature_min_anisotropy = v.cast<double>();
        else if (k == "strokes") {
            for (auto s : v) {
                Eigen::MatrixXd P = s.cast<Eigen::MatrixXd>();
                std::vector<Vector3d> pts;
                for (int i = 0; i < P.rows(); ++i) pts.push_back(P.row(i).transpose());
                p.strokes.push_back(pts);
            }
        } else if (k == "strokes_as_edges") p.strokes_as_edges = v.cast<bool>();
        else if (k == "guide_field") p.guide_field = v.cast<Eigen::MatrixXd>();
        else if (k == "guide_weight") p.guide_weight = v.cast<Eigen::VectorXd>();
        else if (k == "guide_strength") p.guide_strength = v.cast<double>();
        else if (k == "density") p.density = v.cast<Eigen::VectorXd>();
        else if (k == "density_regions") {
            for (auto r : v) {
                auto t = r.cast<py::tuple>();
                DensityRegion d;
                auto c = t[0].cast<std::vector<double>>();
                d.center = Vector3d(c[0], c[1], c[2]);
                d.radius = t[1].cast<double>();
                d.factor = t[2].cast<double>();
                p.density_regions.push_back(d);
            }
        } else if (k == "curvature_density") p.curvature_density = v.cast<double>();
        else if (k == "max_density_ratio") p.max_density_ratio = v.cast<double>();
        else if (k == "symmetry_axis") {
            if (py::isinstance<py::str>(v)) {
                std::string a = v.cast<std::string>();
                p.symmetry_axis = a == "x" ? 0 : a == "y" ? 1 : a == "z" ? 2 : -1;
            } else p.symmetry_axis = v.cast<int>();
        } else if (k == "symmetry_offset") p.symmetry_offset = v.cast<double>();
        else if (k == "max_hole_perimeter") p.max_hole_perimeter = v.cast<double>();
        else if (k == "min_component_area") p.min_component_area = v.cast<double>();
        else if (k == "min_component_faces") p.min_component_faces = v.cast<int>();
        else if (k == "pure_quad") p.pure_quad = v.cast<bool>();
        else if (k == "smooth_iterations") p.smooth_iterations = v.cast<int>();
        else if (k == "transfer_uv") p.transfer_uv = v.cast<bool>();
        else if (k == "seed") p.seed = v.cast<unsigned int>();
        else if (k == "verbose") p.verbose = v.cast<bool>() ? 1 : 0;
        else throw py::key_error("unknown quadremesh parameter: " + k);
    }
    return p;
}

TriMesh make_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const py::object& UV, const py::object& FUV) {
    TriMesh m;
    m.V = V;
    m.F = F;
    if (!UV.is_none() && !FUV.is_none()) {
        m.UV = UV.cast<Eigen::MatrixXd>();
        m.FUV = FUV.cast<Eigen::MatrixXi>();
    }
    return m;
}

}  // namespace

PYBIND11_MODULE(_quadremesh, m) {
    m.doc() = "quadremesh C++ core: quality-driven triangle-to-structured-quad remeshing";
    m.attr("__version__") = version();

    m.def(
        "remesh",
        [](const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, py::object UV, py::object FUV, py::kwargs kw) {
            Params p = params_from_kwargs(kw);
            TriMesh in = make_mesh(V, F, UV, FUV);
            Result r;
            {
                py::gil_scoped_release release;
                r = quadremesh(in, p);
            }
            py::dict d;
            d["V"] = r.mesh.V;
            d["Q"] = r.mesh.Q;
            d["UV"] = r.mesh.UV;
            d["field_V"] = r.field_V;
            d["field_F"] = r.field_F;
            d["field_Q"] = r.field_Q;
            d["field_N"] = r.field_N;
            d["feature_edges"] = r.feature_edges;
            d["log"] = r.log;
            py::dict t;
            t["preprocess"] = r.timing.preprocess;
            t["features"] = r.timing.features;
            t["field"] = r.timing.field;
            t["extraction"] = r.timing.extraction;
            t["postprocess"] = r.timing.postprocess;
            t["total"] = r.timing.total;
            d["timing"] = t;
            d["components"] = py::make_tuple(r.components_in, r.components_out, r.components_fallback);
            return d;
        },
        py::arg("V"), py::arg("F"), py::arg("UV") = py::none(), py::arg("FUV") = py::none(),
        "Remesh (V, F) -> dict with V, Q (k x 4, -1 for triangles), UV and diagnostics.");

    m.def(
        "evaluate",
        [](const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& QV, const Eigen::MatrixXi& Q,
           int target_faces, double feature_angle, double min_feature_length, int samples, double runtime,
           const std::string& method) {
            TriMesh in;
            in.V = V;
            in.F = F;
            QuadMesh out;
            out.V = QV;
            out.Q = Q;
            EvalParams ep;
            ep.target_faces = target_faces;
            ep.feature_angle = feature_angle;
            ep.min_feature_length = min_feature_length;
            ep.samples = samples;
            std::string js;
            {
                py::gil_scoped_release release;
                js = evaluate(in, out, ep, runtime, method);
            }
            return js;
        },
        py::arg("V"), py::arg("F"), py::arg("QV"), py::arg("Q"), py::arg("target_faces") = -1,
        py::arg("feature_angle") = 35.0, py::arg("min_feature_length") = 3.0, py::arg("samples") = 200000,
        py::arg("runtime") = -1.0, py::arg("method") = "", "Quality report (JSON string).");

    m.def(
        "load_mesh",
        [](const std::string& path) {
            TriMesh t = load_mesh(path);
            py::dict d;
            d["V"] = t.V;
            d["F"] = t.F;
            if (t.has_uv()) {
                d["UV"] = t.UV;
                d["FUV"] = t.FUV;
            }
            return d;
        },
        py::arg("path"));
    m.def(
        "load_quad_obj",
        [](const std::string& path) {
            QuadMesh q = load_quad_obj(path);
            return py::make_tuple(q.V, q.Q);
        },
        py::arg("path"));
    m.def(
        "save_quad_obj",
        [](const std::string& path, const Eigen::MatrixXd& V, const Eigen::MatrixXi& Q, py::object UV) {
            QuadMesh q;
            q.V = V;
            q.Q = Q;
            if (!UV.is_none()) q.UV = UV.cast<Eigen::MatrixXd>();
            save_quad_obj(path, q);
        },
        py::arg("path"), py::arg("V"), py::arg("Q"), py::arg("UV") = py::none());
}
