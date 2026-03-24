#include "chemsim/flowsheet/Flowsheet.hpp"
#include <iostream>
#include <memory>

int main() {
    try {
        auto fs = std::make_unique<chemsim::Flowsheet>(
            chemsim::Flowsheet::fromJSON("examples/simple_recycle.json", "data/components.json"));
        std::cout << "loaded\n";
        fs->solve();
        std::cout << "solved\n";
        auto names = fs->streamNames();
        std::cout << "streams=" << names.size() << "\n";
        std::cout << fs->summary() << "\n";
        std::cout << fs->resultsAsJson().dump(2) << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "EX: " << ex.what() << "\n";
        return 2;
    }
}
