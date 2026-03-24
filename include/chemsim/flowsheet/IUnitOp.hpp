#pragma once
#include "chemsim/core/Stream.hpp"
#include <string>
#include <vector>
#include <stdexcept>

namespace chemsim {

/// Abstract interface for any unit operation managed by a Flowsheet.
class IUnitOp {
public:
    virtual ~IUnitOp() = default;

    virtual std::vector<std::string> inletPorts()  const = 0;
    virtual std::vector<std::string> outletPorts() const = 0;

    /// Assign inlet stream by port name (called by flowsheet before solve).
    virtual void setInlet(const std::string& port, const Stream& s) = 0;

    /// Retrieve outlet stream by port name (valid after solve()).
    virtual const Stream& getOutlet(const std::string& port) const = 0;

    /// Execute the unit calculation with the currently assigned inlets.
    virtual void solve() = 0;
};

} // namespace chemsim
