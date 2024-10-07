#include "cdist.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuda implementation of distance matrix";

    py::enum_<distanceType>(m, "distance_type")
        .value("L1", distanceType::L1)
        .value("L2", distanceType::L2)
        .export_values();

    m.def("distance", &distance,
          py::arg("src"),
          py::arg("dst"),
          py::arg("type") = distanceType::L1);
    m.def("fast_distance", &fastDistance,
          py::arg("src"),
          py::arg("dst"),
          py::arg("type") = distanceType::L1);
}
