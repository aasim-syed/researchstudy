#include "quadremesh/io.h"

#include <igl/readOFF.h>
#include <igl/readPLY.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

namespace qrm {

std::string lower_ext(const std::string& path) {
    auto p = path.find_last_of('.');
    if (p == std::string::npos) return "";
    std::string e = path.substr(p + 1);
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e;
}

namespace {

// Fast OBJ reader: v / vt / f with negative indices, polygons fan-triangulated.
TriMesh read_obj(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open " + path);
    std::vector<double> v, vt;
    std::vector<int> f, fuv;
    bool uv_ok = true;
    char line[1 << 16];
    std::vector<int> pi, ti;
    while (std::fgets(line, sizeof(line), fp)) {
        char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t')) {
            double x = 0, y = 0, z = 0;
            char* e = s + 2;
            x = std::strtod(e, &e);
            y = std::strtod(e, &e);
            z = std::strtod(e, &e);
            v.push_back(x), v.push_back(y), v.push_back(z);
        } else if (s[0] == 'v' && s[1] == 't' && (s[2] == ' ' || s[2] == '\t')) {
            char* e = s + 3;
            double a = std::strtod(e, &e), b = std::strtod(e, &e);
            vt.push_back(a), vt.push_back(b);
        } else if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
            pi.clear();
            ti.clear();
            char* e = s + 2;
            while (true) {
                while (*e == ' ' || *e == '\t') ++e;
                if (*e == '\0' || *e == '\n' || *e == '\r') break;
                char* end;
                long a = std::strtol(e, &end, 10);
                if (end == e) break;
                e = end;
                long t = 0;
                if (*e == '/') {
                    ++e;
                    if (*e != '/') {
                        t = std::strtol(e, &end, 10);
                        e = end;
                    }
                    if (*e == '/') {
                        ++e;
                        std::strtol(e, &end, 10);
                        e = end;
                    }
                }
                int nv = (int)(v.size() / 3), nt = (int)(vt.size() / 2);
                pi.push_back(a > 0 ? (int)a - 1 : nv + (int)a);
                ti.push_back(t > 0 ? (int)t - 1 : (t < 0 ? nt + (int)t : -1));
                while (*e && *e != ' ' && *e != '\t' && *e != '\n' && *e != '\r') ++e;
            }
            for (size_t k = 1; k + 1 < pi.size(); ++k) {
                f.push_back(pi[0]), f.push_back(pi[k]), f.push_back(pi[k + 1]);
                fuv.push_back(ti[0]), fuv.push_back(ti[k]), fuv.push_back(ti[k + 1]);
                if (ti[0] < 0 || ti[k] < 0 || ti[k + 1] < 0) uv_ok = false;
            }
        }
    }
    std::fclose(fp);
    TriMesh m;
    m.V = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(v.data(), v.size() / 3, 3);
    m.F = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>(f.data(), f.size() / 3, 3);
    if (uv_ok && !vt.empty() && !f.empty()) {
        m.UV = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>(vt.data(), vt.size() / 2, 2);
        m.FUV = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>(fuv.data(), fuv.size() / 3, 3);
    }
    return m;
}

TriMesh read_gltf(const std::string& path) {
    cgltf_options opt;
    std::memset(&opt, 0, sizeof(opt));
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&opt, path.c_str(), &data) != cgltf_result_success)
        throw std::runtime_error("cannot parse glTF " + path);
    if (cgltf_load_buffers(&opt, data, path.c_str()) != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("cannot load glTF buffers " + path);
    }
    std::vector<double> v, uv;
    std::vector<int> f;
    bool all_uv = true;
    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        cgltf_node* node = &data->nodes[ni];
        if (!node->mesh) continue;
        float M[16];
        cgltf_node_transform_world(node, M);
        Eigen::Matrix4d T;
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) T(r, c) = M[c * 4 + r];
        for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
            cgltf_primitive* prim = &node->mesh->primitives[pi];
            if (prim->type != cgltf_primitive_type_triangles) continue;
            cgltf_accessor *pos = nullptr, *tex = nullptr;
            for (cgltf_size a = 0; a < prim->attributes_count; ++a) {
                if (prim->attributes[a].type == cgltf_attribute_type_position) pos = prim->attributes[a].data;
                if (prim->attributes[a].type == cgltf_attribute_type_texcoord && prim->attributes[a].index == 0)
                    tex = prim->attributes[a].data;
            }
            if (!pos) continue;
            int base = (int)(v.size() / 3);
            for (cgltf_size k = 0; k < pos->count; ++k) {
                float p[3];
                cgltf_accessor_read_float(pos, k, p, 3);
                Eigen::Vector4d q = T * Eigen::Vector4d(p[0], p[1], p[2], 1.0);
                v.push_back(q[0]), v.push_back(q[1]), v.push_back(q[2]);
                float t[2] = {0, 0};
                if (tex) cgltf_accessor_read_float(tex, k, t, 2);
                uv.push_back(t[0]), uv.push_back(1.0 - t[1]);
            }
            if (!tex) all_uv = false;
            if (prim->indices) {
                for (cgltf_size k = 0; k + 2 < prim->indices->count + 0; k += 3) {
                    for (int j = 0; j < 3; ++j) f.push_back(base + (int)cgltf_accessor_read_index(prim->indices, k + j));
                }
            } else {
                for (cgltf_size k = 0; k + 2 < pos->count; k += 3)
                    for (int j = 0; j < 3; ++j) f.push_back(base + (int)(k + j));
            }
        }
    }
    cgltf_free(data);
    TriMesh m;
    m.V = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(v.data(), v.size() / 3, 3);
    m.F = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>(f.data(), f.size() / 3, 3);
    if (all_uv && !uv.empty()) {
        m.UV = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>(uv.data(), uv.size() / 2, 2);
        m.FUV = m.F;
    }
    return m;
}

}  // namespace

TriMesh load_mesh(const std::string& path) {
    std::string ext = lower_ext(path);
    TriMesh m;
    if (ext == "obj") {
        m = read_obj(path);
    } else if (ext == "glb" || ext == "gltf") {
        m = read_gltf(path);
    } else if (ext == "ply") {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        if (!igl::readPLY(path, V, F)) throw std::runtime_error("cannot read PLY " + path);
        m.V = V;
        m.F = F;
    } else if (ext == "off") {
        if (!igl::readOFF(path, m.V, m.F)) throw std::runtime_error("cannot read OFF " + path);
    } else {
        throw std::runtime_error("unsupported mesh format: " + path);
    }
    if (m.F.rows() == 0 || m.F.cols() != 3) throw std::runtime_error("no triangles in " + path);
    // drop faces with out-of-range indices
    int bad = 0;
    for (int i = 0; i < m.F.rows(); ++i)
        for (int j = 0; j < 3; ++j)
            if (m.F(i, j) < 0 || m.F(i, j) >= m.V.rows()) bad++;
    if (bad) {
        std::vector<int> keep;
        for (int i = 0; i < m.F.rows(); ++i) {
            bool ok = true;
            for (int j = 0; j < 3; ++j) ok &= m.F(i, j) >= 0 && m.F(i, j) < m.V.rows();
            if (ok) keep.push_back(i);
        }
        MatrixXi F2(keep.size(), 3), FUV2;
        if (m.has_uv()) FUV2.resize(keep.size(), 3);
        for (size_t k = 0; k < keep.size(); ++k) {
            F2.row(k) = m.F.row(keep[k]);
            if (m.has_uv()) FUV2.row(k) = m.FUV.row(keep[k]);
        }
        m.F = F2;
        if (m.has_uv() || FUV2.rows()) m.FUV = FUV2;
    }
    return m;
}

void save_quad_obj(const std::string& path, const QuadMesh& m) {
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot write " + path);
    std::fprintf(fp, "# quadremesh output: %d vertices, %d faces\n", (int)m.V.rows(), (int)m.Q.rows());
    for (int i = 0; i < m.V.rows(); ++i) std::fprintf(fp, "v %.9g %.9g %.9g\n", m.V(i, 0), m.V(i, 1), m.V(i, 2));
    bool uv = m.has_uv();
    if (uv)
        for (int i = 0; i < m.UV.rows(); ++i) std::fprintf(fp, "vt %.7g %.7g\n", m.UV(i, 0), 1.0 - m.UV(i, 1));
    for (int i = 0; i < m.Q.rows(); ++i) {
        int n = m.Q(i, 3) < 0 ? 3 : 4;
        std::fputc('f', fp);
        for (int j = 0; j < n; ++j) {
            if (uv)
                std::fprintf(fp, " %d/%d", m.Q(i, j) + 1, 4 * i + j + 1);
            else
                std::fprintf(fp, " %d", m.Q(i, j) + 1);
        }
        std::fputc('\n', fp);
    }
    std::fclose(fp);
}

QuadMesh load_quad_obj(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("cannot open " + path);
    std::vector<double> v;
    std::vector<std::array<int, 4>> q;
    char line[1 << 16];
    std::vector<int> pi;
    while (std::fgets(line, sizeof(line), fp)) {
        char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t')) {
            char* e = s + 2;
            double x = std::strtod(e, &e), y = std::strtod(e, &e), z = std::strtod(e, &e);
            v.push_back(x), v.push_back(y), v.push_back(z);
        } else if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
            pi.clear();
            char* e = s + 2;
            while (true) {
                while (*e == ' ' || *e == '\t') ++e;
                if (*e == '\0' || *e == '\n' || *e == '\r') break;
                char* end;
                long a = std::strtol(e, &end, 10);
                if (end == e) break;
                e = end;
                while (*e && *e != ' ' && *e != '\t' && *e != '\n' && *e != '\r') ++e;
                int nv = (int)(v.size() / 3);
                pi.push_back(a > 0 ? (int)a - 1 : nv + (int)a);
            }
            if (pi.size() == 3) q.push_back({pi[0], pi[1], pi[2], -1});
            else if (pi.size() == 4) q.push_back({pi[0], pi[1], pi[2], pi[3]});
            else
                for (size_t k = 1; k + 1 < pi.size(); ++k) q.push_back({pi[0], pi[k], pi[k + 1], -1});
        }
    }
    std::fclose(fp);
    QuadMesh m;
    m.V = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(v.data(), v.size() / 3, 3);
    m.Q.resize(q.size(), 4);
    for (size_t i = 0; i < q.size(); ++i)
        for (int j = 0; j < 4; ++j) m.Q(i, j) = q[i][j];
    return m;
}

void save_tri_obj(const std::string& path, const TriMesh& m) {
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot write " + path);
    for (int i = 0; i < m.V.rows(); ++i) std::fprintf(fp, "v %.9g %.9g %.9g\n", m.V(i, 0), m.V(i, 1), m.V(i, 2));
    for (int i = 0; i < m.F.rows(); ++i) std::fprintf(fp, "f %d %d %d\n", m.F(i, 0) + 1, m.F(i, 1) + 1, m.F(i, 2) + 1);
    std::fclose(fp);
}

}  // namespace qrm
