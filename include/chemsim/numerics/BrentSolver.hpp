#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace chemsim {

class BrentSolver {
public:
    struct Options {
        double tol    = 1e-10;
        int    maxIter = 200;
    };

    struct Result {
        double root;
        double residual;
        int    iterations;
        bool   converged;
    };

    // Solve f(x)=0 on [a,b]. Throws if bracket is invalid or no convergence.
    template<typename F>
    static Result solve(F&& f, double a, double b) {
        return solve(std::forward<F>(f), a, b, Options{});
    }

    template<typename F>
    static Result solve(F&& f, double a, double b, Options opts) {
        double fa = f(a);
        double fb = f(b);

        if (fa * fb > 0.0)
            throw std::domain_error(
                "BrentSolver: f(a) and f(b) have the same sign — no bracket");

        // Exact roots at endpoints
        if (fa == 0.0) return {a, 0.0, 0, true};
        if (fb == 0.0) return {b, 0.0, 0, true};

        // Ensure |f(a)| >= |f(b)|
        if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }

        double c  = a, fc = fa;
        double s  = b, fs = fb;
        double d  = 0.0;
        bool   mflag = true;

        for (int iter = 1; iter <= opts.maxIter; ++iter) {
            if (std::abs(b - a) < opts.tol || std::abs(fs) < opts.tol)
                return {s, fs, iter, true};

            if (fa != fc && fb != fc) {
                // Inverse quadratic interpolation
                s = a*fb*fc/((fa-fb)*(fa-fc))
                  + b*fa*fc/((fb-fa)*(fb-fc))
                  + c*fa*fb/((fc-fa)*(fc-fb));
            } else {
                // Secant
                s = b - fb*(b-a)/(fb-fa);
            }

            // Bisection conditions
            bool cond1 = !((3*a+b)/4 < s && s < b) && !((3*a+b)/4 > s && s > b);
            // simplified: s outside ((3a+b)/4, b)
            double ab4 = (3*a+b)/4.0;
            cond1 = !((s > ab4 && s < b) || (s < ab4 && s > b));
            bool cond2 = mflag  && std::abs(s-b) >= std::abs(b-c)/2.0;
            bool cond3 = !mflag && std::abs(s-b) >= std::abs(c-d)/2.0;
            bool cond4 = mflag  && std::abs(b-c)  < opts.tol;
            bool cond5 = !mflag && std::abs(c-d)  < opts.tol;

            if (cond1 || cond2 || cond3 || cond4 || cond5) {
                s = (a + b) / 2.0;
                mflag = true;
            } else {
                mflag = false;
            }

            fs = f(s);
            d  = c;
            c  = b; fc = fb;

            if (fa * fs < 0.0) { b = s; fb = fs; }
            else               { a = s; fa = fs; }

            if (std::abs(fa) < std::abs(fb)) {
                std::swap(a, b); std::swap(fa, fb);
            }
        }

        throw std::runtime_error(
            "BrentSolver: did not converge in " + std::to_string(opts.maxIter) + " iterations");
    }
};

} // namespace chemsim
