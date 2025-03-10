#include "biest_call.h"

// namespace biest_call{
    // #include <gperftools/profiler.h> // Profiler


// double sum_of_sines(Array &m)
// {
//     auto sines = xt::sin(m); // sines does not actually hold values.
//     return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
// }

void test_single(double a, double b, Array &gamma, Array &f_arr, Array &result)
{
    constexpr int DIM = 3; // dimensions of coordinate space
    const int digits = 10; // number of digits of accuracy requested

    const int nfp = 1, Nt = 70, Np = 20;
    sctl::Vector<double> X(DIM * Nt * Np), F(Nt * Np);
    for (int i = 0; i < Nt; i++)
    { // initialize data X, F
        for (int j = 0; j < Np; j++)
        {
            const double phi = 2 * sctl::const_pi<double>() * i / Nt;
            const double theta = 2 * sctl::const_pi<double>() * j / Np;

            const double R = 1 + 0.25 * sctl::cos<double>(theta);
            const double x = R * sctl::cos<double>(phi);
            const double y = R * a * sctl::sin<double>(phi);
            const double z = 0.25 * sctl::sin<double>(theta);

            X[(0 * Nt + i) * Np + j] = x;
            X[(1 * Nt + i) * Np + j] = y;
            X[(2 * Nt + i) * Np + j] = z;
            F[i * Np + j] = x + y + b * z;
            gamma(i, j, 0) = x;
            gamma(i, j, 1) = y;
            gamma(i, j, 2) = z;
            f_arr(i, j) = F[i * Np + j];
        }
    }
    py::print("Initialization successful.");

    constexpr int KER_DIM0 = 1;                                    // input degrees-of-freedom of kernel
    constexpr int KER_DIM1 = 1;                                    // output degrees-of-freedom of kernel
    biest::FieldPeriodBIOp<double, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
    Surface Svec(1);
    Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
    py::print("Surface built successfully.");
    const auto kernel = biest::Laplace3D<double>::FxU(); // Laplace single-layer kernel function
    py::print("Kernel built successfully.");
    biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
    py::print("Singular integrals built successfully.");
    sctl::Vector<double> U;
    biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
    py::print("Eval successful.");
    for (int i = 0; i < Nt; i++)
    {
        for (int j = 0; j < Np; j++)
        {
            result(i, j) = U[i * Np + j];
        }
    }
}

void test_double(double a, double b, Array &gamma, Array &f_arr, Array &result)
{
    constexpr int DIM = 3; // dimensions of coordinate space
    const int digits = 10; // number of digits of accuracy requested

    const int nfp = 1, Nt = 70, Np = 20;
    sctl::Vector<double> X(DIM * Nt * Np), F(Nt * Np);
    for (int i = 0; i < Nt; i++)
    { // initialize data X, F
        for (int j = 0; j < Np; j++)
        {
            const double phi = 2 * sctl::const_pi<double>() * i / Nt;
            const double theta = 2 * sctl::const_pi<double>() * j / Np;

            const double R = 1 + 0.25 * sctl::cos<double>(theta);
            const double x = R * sctl::cos<double>(phi);
            const double y = R * a * sctl::sin<double>(phi);
            const double z = 0.25 * sctl::sin<double>(theta);

            X[(0 * Nt + i) * Np + j] = x;
            X[(1 * Nt + i) * Np + j] = y;
            X[(2 * Nt + i) * Np + j] = z;
            F[i * Np + j] = x + y + b * z;
            gamma(i, j, 0) = x;
            gamma(i, j, 1) = y;
            gamma(i, j, 2) = z;
            f_arr(i, j) = F[i * Np + j];
        }
    }
    py::print("Initialization successful.");

    constexpr int KER_DIM0 = 1;                                    // input degrees-of-freedom of kernel
    constexpr int KER_DIM1 = 1;                                    // output degrees-of-freedom of kernel
    biest::FieldPeriodBIOp<double, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
    Surface Svec(1);
    Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
    py::print("Surface built successfully.");
    const auto kernel = biest::Laplace3D<double>::DxU(); // Laplace double-layer kernel function
    py::print("Kernel built successfully.");
    biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
    py::print("Singular integrals built successfully.");
    sctl::Vector<double> U;
    biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
    py::print("Eval successful.");
    for (int i = 0; i < Nt; i++)
    {
        for (int j = 0; j < Np; j++)
        {
            result(i, j) = U[i * Np + j];
        }
    }
}

/*
xt::pyarray<double> &gamma:
r(theta, zeta) of the surface, corresponds to Surface.gamma() in simsopt.
Has shape [n_phi, n_theta, 3]

xt::pyarray<double> &func_in, xt::pyarray<double> &func_in_double:
f(theta, zeta), a scalar function to multiply with a single/double layer
Laplacian kernel and integrate over the surface. Has shape [n_phi, n_theta, dim]
dim must be at least [n_phi, n_theta, n].

bool single:
If true, use the single layer kernel 1/|y-x|. Otherwise, uses the double-layer kernel.

int digits:
Number of digits.

int nfp:
Number of field periods.

bool undo_flip: 
BIEST automatically flips handedness when the normal vector is detected to 
point inward. This can be disadvantageous when integrating a vector proportional
to the normal vector using the double layer kernel, because it can break the 
handedness-independence of some results. Enable to add a sign flip when the 
normal is flipped. Only for double-layered kernel.
*/
static void integrate_multi(
    Array &gamma,
    Array &func_in,
    Array &result,
    bool single,
    int digits,
    int nfp)
{
    // Checking shapes
    if (func_in.dimension() != 3)
    {
        throw std::invalid_argument("func_in has invalid shape.");
    }
    if (func_in.shape(0) != result.shape(0) || func_in.shape(1) != result.shape(1) || func_in.shape(2) != result.shape(2))
    {
        throw std::invalid_argument("func_in and result has different shapes.");
    }
    if (func_in.shape(0) != gamma.shape(0) || func_in.shape(1) != gamma.shape(1) || gamma.shape(2) != 3)
    {
        throw std::invalid_argument("gamma has invalid shape.");
    }
    const int Nvec = func_in.shape(2);
    const int Nt = func_in.shape(0);
    const int Np = func_in.shape(1);

    // Loading gamma
    sctl::Vector<double> X(DIM * Nt * Np);
    for (int i = 0; i < Nt; i++)
    { // initialize data X
        for (int j = 0; j < Np; j++)
        {
            X[(0 * Nt + i) * Np + j] = gamma(i, j, 0); // x
            X[(1 * Nt + i) * Np + j] = gamma(i, j, 1); // y
            X[(2 * Nt + i) * Np + j] = gamma(i, j, 2); // z
        }
    }
    // Constructing the surface.
    biest::FieldPeriodBIOp<double, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
    Surface Svec(1);
    Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
    if (single)
    {
        const auto kernel = biest::Laplace3D<double>::FxU();             // Laplace single-layer kernel function
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
#pragma omp parallel for
        for (int k = 0; k < Nvec; k++)
        {
            sctl::Vector<double> F(Nt * Np), U;
            for (int i = 0; i < Nt; i++)
            {
                for (int j = 0; j < Np; j++)
                {
                    F[i * Np + j] = func_in(i, j, k);
                }
            }
            biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
            for (int i = 0; i < Nt; i++)
            {
                for (int j = 0; j < Np; j++)
                {
                    result(i, j, k) = U[i * Np + j];
                }
            }
        }
    }
    else
    {
        // Constructing kernel and setting up the integral
        const auto kernel = biest::Laplace3D<double>::DxU();             // Laplace double-layer kernel function
        biop.SetupSingular(Svec, kernel, digits, nfp, Nt, Np, Nt, Np); // initialize biop
#pragma omp parallel for
        for (int k = 0; k < Nvec; k++)
        {
            sctl::Vector<double> F(Nt * Np), U;
            for (int i = 0; i < Nt; i++)
            {
                for (int j = 0; j < Np; j++)
                {
                    F[i * Np + j] = func_in(i, j, k);
                }
            }
            biop.Eval(U, F, nfp, Nt, Np); // evaluate potential
            for (int i = 0; i < Nt; i++)
            {
                for (int j = 0; j < Np; j++)
                {
                    // Related to sign flip detection
                    // result(i, j, k) = sign * U[i * Np + j];
                    result(i, j, k) = U[i * Np + j];
                }
            }
        }
    }
}

static void plot_in_vtk(Array &gamma, int digits, int nfp)
{
    const int Nt = gamma.shape(0);
    const int Np = gamma.shape(1);
    // Loading gamma
    sctl::Vector<double> X(DIM * Nt * Np);
    for (int i = 0; i < Nt; i++)
    { // initialize data X
        for (int j = 0; j < Np; j++)
        {
            X[(0 * Nt + i) * Np + j] = gamma(i, j, 0); // x
            X[(1 * Nt + i) * Np + j] = gamma(i, j, 1); // y
            X[(2 * Nt + i) * Np + j] = gamma(i, j, 2); // z
        }
    }
    // Constructing the surface.
    biest::FieldPeriodBIOp<double, DIM, KER_DIM0, KER_DIM1, 0> biop; // boundary integral operator
    Surface Svec(1);
    Svec[0] = biop.BuildSurface(X, nfp, Nt, Np); // build surface object
    WriteVTK("S", Svec); // Write to file for visualization
}

// PYBIND11_MODULE(biest_call, m)
// {
//     xt::import_numpy();
//     m.doc() = "Test module for xtensor python bindings";
//     m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
//     m.def("test_single", test_single, "Testing 100 BIEST calls.");
//     m.def("test_double", test_double, "Testing 100 BIEST calls.");
//     m.def("integrate_multi", integrate_multi, "Integrating multiple scalar functions using BIEST");
//     m.def("plot_in_vtk", plot_in_vtk, "Integrating multiple scalar functions using BIEST");
// }
// }
