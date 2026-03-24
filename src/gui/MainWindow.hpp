#pragma once
#include "chemsim/flowsheet/Flowsheet.hpp"
#include <QMainWindow>
#include <memory>

class FlowsheetCanvas;
class QListWidget;
class QTextEdit;
class QTreeWidget;

class MainWindow : public QMainWindow {
public:
    MainWindow();

private:
    void buildChrome();
    void loadExampleFlowsheet();
    void refreshViews();
    void showInspectorText(const QString& text);

    FlowsheetCanvas* canvas_{};
    QListWidget* palette_{};
    QTreeWidget* streamTree_{};
    QTextEdit* inspector_{};
    QTextEdit* console_{};
    std::unique_ptr<chemsim::Flowsheet> flowsheet_;
};
