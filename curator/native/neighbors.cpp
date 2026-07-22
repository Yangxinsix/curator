#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Vec3 {
    double x;
    double y;
    double z;
};

struct Mat3 {
    double m[3][3];
};

struct BinKey {
    int x;
    int y;
    int z;

    bool operator==(const BinKey& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct BinKeyHash {
    std::size_t operator()(const BinKey& key) const noexcept {
        std::size_t h = static_cast<std::size_t>(key.x) * 73856093u;
        h ^= static_cast<std::size_t>(key.y) * 19349663u;
        h ^= static_cast<std::size_t>(key.z) * 83492791u;
        return h;
    }
};

struct PairRecord {
    int32_t i;
    int32_t j;
    int8_t sx;
    int8_t sy;
    int8_t sz;
    double distance;
};

class BufferView {
public:
    BufferView(PyObject* obj, const char* name) : name_(name) {
        if (PyObject_GetBuffer(obj, &view_, PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) {
            throw std::runtime_error(std::string("failed to acquire contiguous buffer for ") + name_);
        }
        active_ = true;
    }

    ~BufferView() {
        if (active_) {
            PyBuffer_Release(&view_);
        }
    }

    BufferView(const BufferView&) = delete;
    BufferView& operator=(const BufferView&) = delete;

    int ndim() const {
        return view_.ndim;
    }

    Py_ssize_t dim(int index) const {
        return view_.shape[index];
    }

    void require_itemsize(Py_ssize_t itemsize, const char* dtype_name) const {
        if (view_.itemsize != itemsize) {
            throw std::runtime_error(std::string(name_) + " must have dtype " + dtype_name);
        }
    }

    template <typename T>
    const T* data() const {
        return static_cast<const T*>(view_.buf);
    }

private:
    Py_buffer view_{};
    const char* name_;
    bool active_ = false;
};

class GilRelease {
public:
    GilRelease() : state_(PyEval_SaveThread()) {}
    ~GilRelease() {
        PyEval_RestoreThread(state_);
    }

    GilRelease(const GilRelease&) = delete;
    GilRelease& operator=(const GilRelease&) = delete;

private:
    PyThreadState* state_;
};

bool pair_record_less(const PairRecord& left, const PairRecord& right) {
    if (left.i != right.i) {
        return left.i < right.i;
    }
    if (left.j != right.j) {
        return left.j < right.j;
    }
    if (left.sx != right.sx) {
        return left.sx < right.sx;
    }
    if (left.sy != right.sy) {
        return left.sy < right.sy;
    }
    if (left.sz != right.sz) {
        return left.sz < right.sz;
    }
    return left.distance < right.distance;
}

int resolve_thread_count(int requested, std::size_t work_items, std::size_t chunk_size) {
    if (requested < 0) {
        throw std::runtime_error("num_threads must be non-negative");
    }
    if (work_items == 0) {
        return 1;
    }

    const std::size_t chunk_count = (work_items + chunk_size - 1) / chunk_size;
    if (requested > 0) {
        return static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(requested),
            chunk_count
        ));
    }

    // Spawning threads for small neighbor searches costs more than it saves.
    // Keep auto mode conservative and bounded so that multiple data-loader
    // workers cannot each fan out across every logical CPU.
    constexpr std::size_t min_parallel_work_items = 20000;
    constexpr std::size_t target_work_items_per_thread = 8192;
    constexpr std::size_t max_auto_threads = 8;
    if (work_items < min_parallel_work_items) {
        return 1;
    }

    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware == 0) {
        hardware = 1;
    }
    const std::size_t useful_threads =
        (work_items + target_work_items_per_thread - 1) / target_work_items_per_thread;
    const std::size_t threads = std::min({
        static_cast<std::size_t>(hardware),
        max_auto_threads,
        useful_threads,
        chunk_count,
    });
    return static_cast<int>(std::max<std::size_t>(1, threads));
}

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double norm(const Vec3& a) {
    return std::sqrt(dot(a, a));
}

Vec3 row(const Mat3& cell, int index) {
    return Vec3{cell.m[index][0], cell.m[index][1], cell.m[index][2]};
}

Vec3 add(const Vec3& a, const Vec3& b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 sub(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 scale(const Vec3& a, double value) {
    return Vec3{a.x * value, a.y * value, a.z * value};
}

Vec3 matmul_row(const Vec3& v, const Mat3& matrix) {
    return Vec3{
        v.x * matrix.m[0][0] + v.y * matrix.m[1][0] + v.z * matrix.m[2][0],
        v.x * matrix.m[0][1] + v.y * matrix.m[1][1] + v.z * matrix.m[2][1],
        v.x * matrix.m[0][2] + v.y * matrix.m[1][2] + v.z * matrix.m[2][2],
    };
}

double determinant(const Mat3& matrix) {
    const double a = matrix.m[0][0];
    const double b = matrix.m[0][1];
    const double c = matrix.m[0][2];
    const double d = matrix.m[1][0];
    const double e = matrix.m[1][1];
    const double f = matrix.m[1][2];
    const double g = matrix.m[2][0];
    const double h = matrix.m[2][1];
    const double i = matrix.m[2][2];
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

Mat3 inverse(const Mat3& matrix) {
    const double det = determinant(matrix);
    if (std::abs(det) <= 1.0e-14) {
        throw std::runtime_error("cell matrix is singular");
    }
    Mat3 out{};
    out.m[0][0] = (matrix.m[1][1] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][1]) / det;
    out.m[0][1] = (matrix.m[0][2] * matrix.m[2][1] - matrix.m[0][1] * matrix.m[2][2]) / det;
    out.m[0][2] = (matrix.m[0][1] * matrix.m[1][2] - matrix.m[0][2] * matrix.m[1][1]) / det;
    out.m[1][0] = (matrix.m[1][2] * matrix.m[2][0] - matrix.m[1][0] * matrix.m[2][2]) / det;
    out.m[1][1] = (matrix.m[0][0] * matrix.m[2][2] - matrix.m[0][2] * matrix.m[2][0]) / det;
    out.m[1][2] = (matrix.m[0][2] * matrix.m[1][0] - matrix.m[0][0] * matrix.m[1][2]) / det;
    out.m[2][0] = (matrix.m[1][0] * matrix.m[2][1] - matrix.m[1][1] * matrix.m[2][0]) / det;
    out.m[2][1] = (matrix.m[0][1] * matrix.m[2][0] - matrix.m[0][0] * matrix.m[2][1]) / det;
    out.m[2][2] = (matrix.m[0][0] * matrix.m[1][1] - matrix.m[0][1] * matrix.m[1][0]) / det;
    return out;
}

Mat3 identity_matrix() {
    Mat3 out{};
    out.m[0][0] = 1.0;
    out.m[1][1] = 1.0;
    out.m[2][2] = 1.0;
    return out;
}

double cell_height(const Mat3& cell, int axis) {
    const Vec3 a = row(cell, 0);
    const Vec3 b = row(cell, 1);
    const Vec3 c = row(cell, 2);
    const double volume = std::abs(dot(a, cross(b, c)));
    Vec3 face{};
    if (axis == 0) {
        face = cross(b, c);
    } else if (axis == 1) {
        face = cross(c, a);
    } else {
        face = cross(a, b);
    }
    const double area = norm(face);
    if (area <= 1.0e-14) {
        return 0.0;
    }
    return volume / area;
}

bool offset_is_zero(int sx, int sy, int sz) {
    return sx == 0 && sy == 0 && sz == 0;
}

bool offset_is_positive(int sx, int sy, int sz) {
    return sx > 0 || (sx == 0 && sy > 0) || (sx == 0 && sy == 0 && sz > 0);
}

int bin_coord(double value, double origin, double bin_size) {
    return static_cast<int>(std::floor((value - origin) / bin_size));
}

void require_2d_shape(const BufferView& buffer, Py_ssize_t cols, const char* name) {
    if (buffer.ndim() != 2 || buffer.dim(1) != cols) {
        throw std::runtime_error(std::string(name) + " must have shape (N, " + std::to_string(cols) + ")");
    }
}

bool read_bool_sequence(PyObject* obj, int index) {
    PyObject* item = PySequence_GetItem(obj, index);
    if (item == nullptr) {
        throw std::runtime_error("pbc must have length 3");
    }
    const int truth = PyObject_IsTrue(item);
    Py_DECREF(item);
    if (truth < 0) {
        throw std::runtime_error("pbc values must be truthy/falsy");
    }
    return truth != 0;
}

PyObject* make_bytearray_int32(const std::vector<PairRecord>& records, int field) {
    PyObject* obj = PyByteArray_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(records.size() * sizeof(int32_t)));
    if (obj == nullptr) {
        return nullptr;
    }
    auto* data = reinterpret_cast<int32_t*>(PyByteArray_AS_STRING(obj));
    for (std::size_t index = 0; index < records.size(); ++index) {
        data[index] = field == 0 ? records[index].i : records[index].j;
    }
    return obj;
}

PyObject* make_bytearray_int8(const std::vector<PairRecord>& records, int field) {
    PyObject* obj = PyByteArray_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(records.size() * sizeof(int8_t)));
    if (obj == nullptr) {
        return nullptr;
    }
    auto* data = reinterpret_cast<int8_t*>(PyByteArray_AS_STRING(obj));
    for (std::size_t index = 0; index < records.size(); ++index) {
        data[index] = field == 0 ? records[index].sx : (field == 1 ? records[index].sy : records[index].sz);
    }
    return obj;
}

PyObject* make_bytearray_float64(const std::vector<PairRecord>& records) {
    PyObject* obj = PyByteArray_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(records.size() * sizeof(double)));
    if (obj == nullptr) {
        return nullptr;
    }
    auto* data = reinterpret_cast<double*>(PyByteArray_AS_STRING(obj));
    for (std::size_t index = 0; index < records.size(); ++index) {
        data[index] = records[index].distance;
    }
    return obj;
}

int add_dict_item(PyObject* dict, const char* key, PyObject* value) {
    if (value == nullptr) {
        return -1;
    }
    const int result = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return result;
}

PyObject* py_neighbor_pairs(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* positions_obj = nullptr;
    PyObject* cell_obj = Py_None;
    PyObject* pbc_obj = Py_None;
    PyObject* atomic_numbers_obj = Py_None;
    PyObject* pair_cutoffs_obj = Py_None;
    double cutoff = 0.0;
    int num_threads_requested = 0;

    static char const* keywords[] = {
        "positions",
        "cutoff",
        "cell",
        "pbc",
        "atomic_numbers",
        "pair_cutoffs",
        "num_threads",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Od|OOOOi",
            const_cast<char**>(keywords),
            &positions_obj,
            &cutoff,
            &cell_obj,
            &pbc_obj,
            &atomic_numbers_obj,
            &pair_cutoffs_obj,
            &num_threads_requested)) {
        return nullptr;
    }

    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "cutoff must be a positive finite number");
        return nullptr;
    }

    try {
        BufferView positions_buffer(positions_obj, "positions");
        positions_buffer.require_itemsize(static_cast<Py_ssize_t>(sizeof(double)), "float64");
        require_2d_shape(positions_buffer, 3, "positions");
        const Py_ssize_t count = positions_buffer.dim(0);
        if (count > static_cast<Py_ssize_t>(INT32_MAX)) {
            throw std::runtime_error("positions has too many rows for int32 neighbor indices");
        }
        const double* pos_data = positions_buffer.data<double>();

        Mat3 cell = identity_matrix();
        std::unique_ptr<BufferView> cell_buffer;
        if (cell_obj != Py_None) {
            cell_buffer = std::make_unique<BufferView>(cell_obj, "cell");
            cell_buffer->require_itemsize(static_cast<Py_ssize_t>(sizeof(double)), "float64");
            if (cell_buffer->ndim() != 2 || cell_buffer->dim(0) != 3 || cell_buffer->dim(1) != 3) {
                throw std::runtime_error("cell must have shape (3, 3)");
            }
            const double* cell_data = cell_buffer->data<double>();
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    cell.m[r][c] = cell_data[r * 3 + c];
                }
            }
        }

        bool pbc[3] = {false, false, false};
        if (pbc_obj != Py_None) {
            if (!PySequence_Check(pbc_obj) || PySequence_Size(pbc_obj) != 3) {
                throw std::runtime_error("pbc must be a length-3 sequence");
            }
            pbc[0] = read_bool_sequence(pbc_obj, 0);
            pbc[1] = read_bool_sequence(pbc_obj, 1);
            pbc[2] = read_bool_sequence(pbc_obj, 2);
        }

        const bool any_pbc = pbc[0] || pbc[1] || pbc[2];
        Mat3 inv_cell = identity_matrix();
        if (any_pbc) {
            inv_cell = inverse(cell);
        }

        const int32_t* atomic_numbers = nullptr;
        std::unique_ptr<BufferView> atomic_numbers_buffer;
        if (atomic_numbers_obj != Py_None) {
            atomic_numbers_buffer = std::make_unique<BufferView>(atomic_numbers_obj, "atomic_numbers");
            atomic_numbers_buffer->require_itemsize(static_cast<Py_ssize_t>(sizeof(int32_t)), "int32");
            if (atomic_numbers_buffer->ndim() != 1 || atomic_numbers_buffer->dim(0) != count) {
                throw std::runtime_error("atomic_numbers must have shape (N,)");
            }
            atomic_numbers = atomic_numbers_buffer->data<int32_t>();
        }

        const double* pair_cutoffs = nullptr;
        Py_ssize_t pair_cutoff_rows = 0;
        Py_ssize_t pair_cutoff_cols = 0;
        std::unique_ptr<BufferView> pair_cutoffs_buffer;
        if (pair_cutoffs_obj != Py_None) {
            if (atomic_numbers == nullptr) {
                throw std::runtime_error("pair_cutoffs requires atomic_numbers");
            }
            pair_cutoffs_buffer = std::make_unique<BufferView>(pair_cutoffs_obj, "pair_cutoffs");
            pair_cutoffs_buffer->require_itemsize(static_cast<Py_ssize_t>(sizeof(double)), "float64");
            if (pair_cutoffs_buffer->ndim() != 2) {
                throw std::runtime_error("pair_cutoffs must have shape (Z, Z)");
            }
            pair_cutoff_rows = pair_cutoffs_buffer->dim(0);
            pair_cutoff_cols = pair_cutoffs_buffer->dim(1);
            pair_cutoffs = pair_cutoffs_buffer->data<double>();
        }

        std::vector<Vec3> positions(static_cast<std::size_t>(count));
        Vec3 bounds_min{0.0, 0.0, 0.0};
        Vec3 bounds_max{0.0, 0.0, 0.0};
        for (Py_ssize_t index = 0; index < count; ++index) {
            Vec3 p{pos_data[index * 3], pos_data[index * 3 + 1], pos_data[index * 3 + 2]};
            if (any_pbc) {
                Vec3 frac = matmul_row(p, inv_cell);
                if (pbc[0]) {
                    frac.x -= std::floor(frac.x);
                }
                if (pbc[1]) {
                    frac.y -= std::floor(frac.y);
                }
                if (pbc[2]) {
                    frac.z -= std::floor(frac.z);
                }
                p = matmul_row(frac, cell);
            }
            positions[static_cast<std::size_t>(index)] = p;
            if (index == 0) {
                bounds_min = p;
                bounds_max = p;
            } else {
                bounds_min.x = std::min(bounds_min.x, p.x);
                bounds_min.y = std::min(bounds_min.y, p.y);
                bounds_min.z = std::min(bounds_min.z, p.z);
                bounds_max.x = std::max(bounds_max.x, p.x);
                bounds_max.y = std::max(bounds_max.y, p.y);
                bounds_max.z = std::max(bounds_max.z, p.z);
            }
        }

        const double bin_size = cutoff;
        const Vec3 origin{bounds_min.x - cutoff, bounds_min.y - cutoff, bounds_min.z - cutoff};
        std::unordered_map<BinKey, std::vector<int32_t>, BinKeyHash> bins;
        bins.reserve(static_cast<std::size_t>(count) * 2 + 1);
        for (Py_ssize_t index = 0; index < count; ++index) {
            const Vec3 p = positions[static_cast<std::size_t>(index)];
            bins[BinKey{bin_coord(p.x, origin.x, bin_size), bin_coord(p.y, origin.y, bin_size), bin_coord(p.z, origin.z, bin_size)}]
                .push_back(static_cast<int32_t>(index));
        }

        int max_offsets[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; ++axis) {
            if (!pbc[axis]) {
                continue;
            }
            const double height = cell_height(cell, axis);
            if (height <= 1.0e-14) {
                throw std::runtime_error("periodic cell axis is singular");
            }
            max_offsets[axis] = std::max(1, static_cast<int>(std::ceil(cutoff / height)));
            if (max_offsets[axis] > 127) {
                throw std::runtime_error("periodic offset exceeds int8 range");
            }
        }

        std::vector<std::tuple<int, int, int>> offsets;
        for (int sx = -max_offsets[0]; sx <= max_offsets[0]; ++sx) {
            for (int sy = -max_offsets[1]; sy <= max_offsets[1]; ++sy) {
                for (int sz = -max_offsets[2]; sz <= max_offsets[2]; ++sz) {
                    if (offset_is_zero(sx, sy, sz) || offset_is_positive(sx, sy, sz)) {
                        offsets.emplace_back(sx, sy, sz);
                    }
                }
            }
        }

        const Vec3 avec = row(cell, 0);
        const Vec3 bvec = row(cell, 1);
        const Vec3 cvec = row(cell, 2);

        const std::size_t total_work = offsets.size() * static_cast<std::size_t>(count);
        const std::size_t chunk_size = std::max<std::size_t>(
            1024,
            static_cast<std::size_t>(count) / 4
        );
        const int thread_count = resolve_thread_count(
            num_threads_requested,
            total_work,
            chunk_size
        );
        std::vector<std::vector<PairRecord>> local_records(static_cast<std::size_t>(thread_count));

        auto scan_range = [&](std::size_t begin, std::size_t end, int thread_index) {
            auto& records = local_records[static_cast<std::size_t>(thread_index)];
            records.reserve(std::max<std::size_t>(1024, (end - begin) / 2));
            for (std::size_t work_index = begin; work_index < end; ++work_index) {
                const std::size_t offset_index = work_index / static_cast<std::size_t>(count);
                const Py_ssize_t j_index = static_cast<Py_ssize_t>(work_index % static_cast<std::size_t>(count));
                const auto& offset = offsets[offset_index];
                const int sx = std::get<0>(offset);
                const int sy = std::get<1>(offset);
                const int sz = std::get<2>(offset);
                const bool zero_offset = offset_is_zero(sx, sy, sz);
                const Vec3 shift = add(add(scale(avec, sx), scale(bvec, sy)), scale(cvec, sz));
                const Vec3 shifted = add(positions[static_cast<std::size_t>(j_index)], shift);
                if (
                    shifted.x < bounds_min.x - cutoff ||
                    shifted.x > bounds_max.x + cutoff ||
                    shifted.y < bounds_min.y - cutoff ||
                    shifted.y > bounds_max.y + cutoff ||
                    shifted.z < bounds_min.z - cutoff ||
                    shifted.z > bounds_max.z + cutoff) {
                    continue;
                }
                const int x0 = bin_coord(shifted.x - cutoff, origin.x, bin_size);
                const int x1 = bin_coord(shifted.x + cutoff, origin.x, bin_size);
                const int y0 = bin_coord(shifted.y - cutoff, origin.y, bin_size);
                const int y1 = bin_coord(shifted.y + cutoff, origin.y, bin_size);
                const int z0 = bin_coord(shifted.z - cutoff, origin.z, bin_size);
                const int z1 = bin_coord(shifted.z + cutoff, origin.z, bin_size);
                for (int bx = x0; bx <= x1; ++bx) {
                    for (int by = y0; by <= y1; ++by) {
                        for (int bz = z0; bz <= z1; ++bz) {
                            auto found = bins.find(BinKey{bx, by, bz});
                            if (found == bins.end()) {
                                continue;
                            }
                            for (int32_t i_index : found->second) {
                                if (zero_offset) {
                                    if (i_index >= j_index) {
                                        continue;
                                    }
                                }

                                double local_cutoff = cutoff;
                                if (pair_cutoffs != nullptr) {
                                    const int32_t zi = atomic_numbers[i_index];
                                    const int32_t zj = atomic_numbers[j_index];
                                    if (zi < 0 || zj < 0 || zi >= pair_cutoff_rows || zj >= pair_cutoff_cols) {
                                        continue;
                                    }
                                    local_cutoff = pair_cutoffs[static_cast<Py_ssize_t>(zi) * pair_cutoff_cols + static_cast<Py_ssize_t>(zj)];
                                    if (!std::isfinite(local_cutoff) || local_cutoff <= 0.0) {
                                        continue;
                                    }
                                    local_cutoff = std::min(local_cutoff, cutoff);
                                }

                                const Vec3 delta = sub(shifted, positions[static_cast<std::size_t>(i_index)]);
                                const double d2 = dot(delta, delta);
                                const double local_cutoff2 = local_cutoff * local_cutoff;
                                if (d2 <= local_cutoff2 + 1.0e-12 && d2 > 1.0e-24) {
                                    records.push_back(PairRecord{
                                        i_index,
                                        static_cast<int32_t>(j_index),
                                        static_cast<int8_t>(sx),
                                        static_cast<int8_t>(sy),
                                        static_cast<int8_t>(sz),
                                        std::sqrt(d2),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        };

        {
            GilRelease release;
            if (thread_count == 1) {
                scan_range(0, total_work, 0);
            } else {
                std::atomic<std::size_t> next_work{0};
                std::vector<std::thread> threads;
                threads.reserve(static_cast<std::size_t>(thread_count));
                for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
                    threads.emplace_back([&, thread_index]() {
                        while (true) {
                            const std::size_t begin = next_work.fetch_add(chunk_size);
                            if (begin >= total_work) {
                                break;
                            }
                            const std::size_t end = std::min(total_work, begin + chunk_size);
                            scan_range(begin, end, thread_index);
                        }
                    });
                }
                for (auto& thread : threads) {
                    thread.join();
                }
            }
        }

        std::size_t record_count = 0;
        for (const auto& chunk : local_records) {
            record_count += chunk.size();
        }
        std::vector<PairRecord> records;
        records.reserve(record_count);
        for (auto& chunk : local_records) {
            records.insert(records.end(), chunk.begin(), chunk.end());
        }
        std::sort(records.begin(), records.end(), pair_record_less);

        PyObject* out = PyDict_New();
        if (out == nullptr) {
            return nullptr;
        }
        if (
            add_dict_item(out, "i", make_bytearray_int32(records, 0)) < 0 ||
            add_dict_item(out, "j", make_bytearray_int32(records, 1)) < 0 ||
            add_dict_item(out, "sx", make_bytearray_int8(records, 0)) < 0 ||
            add_dict_item(out, "sy", make_bytearray_int8(records, 1)) < 0 ||
            add_dict_item(out, "sz", make_bytearray_int8(records, 2)) < 0 ||
            add_dict_item(out, "distance", make_bytearray_float64(records)) < 0) {
            Py_DECREF(out);
            return nullptr;
        }
        return out;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {
        "neighbor_pairs",
        reinterpret_cast<PyCFunction>(py_neighbor_pairs),
        METH_VARARGS | METH_KEYWORDS,
        "Find neighbor pairs with optional periodic offsets.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_neighbors",
    "Native neighbor search kernels for CURATOR.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__neighbors() {
    return PyModule_Create(&module);
}
