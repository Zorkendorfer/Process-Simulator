#include "MainWindow.hpp"
#include "FlowsheetCanvas.hpp"
#include "chemsim/flowsheet/Flowsheet.hpp"
#include <QApplication>
#include <QDockWidget>
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QHeaderView>
#include <QLabel>
#include <QListWidget>
#include <QMenuBar>
#include <QMessageBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTimer>
#include <QTextEdit>
#include <QTextStream>
#include <QToolBar>
#include <QToolButton>
#include <QTreeWidget>
#include <QVBoxLayout>

#ifndef CHEMSIM_DATA_DIR
#define CHEMSIM_DATA_DIR "data"
#endif

namespace {

QString appStyleSheet() {
    return QString::fromUtf8(R"(
QMainWindow, QWidget {
    background: #0b1220;
    color: #dbe3f0;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMenuBar, QToolBar {
    background: #111a2e;
    border: none;
    spacing: 8px;
}
QToolBar {
    padding: 8px 12px;
    border-bottom: 1px solid #1d2940;
}
QToolButton {
    background: #16233b;
    color: #eef4ff;
    border: 1px solid #243553;
    border-radius: 8px;
    padding: 8px 14px;
}
QToolButton:hover {
    background: #1c2d49;
}
QDockWidget::title {
    background: #111a2e;
    padding: 8px 10px;
    border-bottom: 1px solid #1d2940;
    text-transform: uppercase;
    font-weight: 600;
    color: #9fb4d9;
}
QListWidget, QTreeWidget, QTextEdit {
    background: #101a2d;
    border: 1px solid #1f2d46;
    border-radius: 8px;
    selection-background-color: #20416b;
}
QStatusBar {
    background: #0f172a;
    color: #7c93b8;
}
)");
}

QString projectRoot() {
    return QCoreApplication::applicationDirPath() + "/..";
}

void appendGuiLog(const QString& message) {
    QFile file(QDir::cleanPath(QCoreApplication::applicationDirPath() + "/../gui-debug.log"));
    if (!file.open(QIODevice::Append | QIODevice::Text)) {
        return;
    }

    QTextStream out(&file);
    out << message << '\n';
}

} // namespace

MainWindow::MainWindow() {
    setWindowTitle("ChemSim Studio");
    resize(1600, 980);
    setStyleSheet(appStyleSheet());

    buildChrome();
    refreshViews();
    statusBar()->showMessage("Ready");
}

void MainWindow::buildChrome() {
    auto* top_bar = addToolBar("Workspace");
    top_bar->setMovable(false);

    auto* load_button = new QToolButton(this);
    load_button->setText("Load Example");
    top_bar->addWidget(load_button);
    connect(load_button, &QToolButton::clicked, this, [this]() {
        try {
            appendGuiLog("Load Example: begin");
            loadExampleFlowsheet();
            appendGuiLog("Load Example: parsed");
            refreshViews();
            appendGuiLog("Load Example: refreshed");
            statusBar()->showMessage("Loaded recycle example", 3000);
        } catch (const std::exception& ex) {
            const QString message = QString("Load error:\n%1").arg(ex.what());
            appendGuiLog("Load Example: exception");
            appendGuiLog(message);
            console_->setPlainText(message);
            QMessageBox::critical(this, "ChemSim Studio", message);
        } catch (...) {
            const QString message = "Load error: unknown exception";
            appendGuiLog("Load Example: unknown exception");
            console_->setPlainText(message);
            QMessageBox::critical(this, "ChemSim Studio", message);
        }
    });

    auto* solve_button = new QToolButton(this);
    solve_button->setText("Solve");
    top_bar->addWidget(solve_button);
    connect(solve_button, &QToolButton::clicked, this, [this]() {
        try {
            if (!flowsheet_) {
                return;
            }
            appendGuiLog("Solve: begin");
            flowsheet_->solve();
            appendGuiLog("Solve: solved");
            refreshViews();
            appendGuiLog("Solve: refreshed");
            statusBar()->showMessage("Flowsheet solved", 3000);
        } catch (const std::exception& ex) {
            const QString message = QString("Solve error:\n%1").arg(ex.what());
            appendGuiLog("Solve: exception");
            appendGuiLog(message);
            console_->setPlainText(message);
            QMessageBox::critical(this, "ChemSim Studio", message);
        } catch (...) {
            const QString message = "Solve error: unknown exception";
            appendGuiLog("Solve: unknown exception");
            console_->setPlainText(message);
            QMessageBox::critical(this, "ChemSim Studio", message);
        }
    });

    auto* export_button = new QToolButton(this);
    export_button->setText("Export Results");
    top_bar->addWidget(export_button);
    connect(export_button, &QToolButton::clicked, this, [this]() {
        if (flowsheet_) {
            const auto path = projectRoot() + "/gui-results.json";
            flowsheet_->exportResults(path.toStdString());
            statusBar()->showMessage("Exported results to gui-results.json", 4000);
        }
    });

    canvas_ = new FlowsheetCanvas(this);
    canvas_->populateDemoDiagram();
    setCentralWidget(canvas_);

    auto* palette_dock = new QDockWidget("Equipment Palette", this);
    palette_ = new QListWidget(palette_dock);
    palette_->addItems({"Material Stream", "Mixer", "Flash Drum", "Pump", "Splitter", "Heat Exchanger", "Reactor", "Column"});
    palette_dock->setWidget(palette_);
    addDockWidget(Qt::LeftDockWidgetArea, palette_dock);

    auto* right_split = new QSplitter(Qt::Vertical, this);

    streamTree_ = new QTreeWidget(right_split);
    streamTree_->setHeaderLabels({"Stream", "Flow (mol/s)", "Phase"});
    streamTree_->header()->setSectionResizeMode(QHeaderView::Stretch);

    inspector_ = new QTextEdit(right_split);
    inspector_->setReadOnly(true);

    right_split->setStretchFactor(0, 3);
    right_split->setStretchFactor(1, 2);

    auto* inspect_dock = new QDockWidget("Inspector", this);
    inspect_dock->setWidget(right_split);
    addDockWidget(Qt::RightDockWidgetArea, inspect_dock);

    console_ = new QTextEdit(this);
    console_->setReadOnly(true);
    console_->setMaximumHeight(180);

    auto* console_dock = new QDockWidget("Messages", this);
    console_dock->setWidget(console_);
    addDockWidget(Qt::BottomDockWidgetArea, console_dock);
}

void MainWindow::loadExampleFlowsheet() {
    const QString base = QDir::cleanPath(QCoreApplication::applicationDirPath() + "/..");
    const QString flowsheet_path = base + "/examples/simple_recycle.json";
    const QString component_db = base + "/data/components.json";

    if (!QFileInfo::exists(flowsheet_path) || !QFileInfo::exists(component_db)) {
        throw std::runtime_error("Unable to locate example JSON or component database");
    }

    flowsheet_ = chemsim::Flowsheet::fromJSONUnique(
        flowsheet_path.toStdString(),
        component_db.toStdString());
}

void MainWindow::refreshViews() {
    streamTree_->clear();
    if (!flowsheet_) {
        inspector_->setPlainText("No flowsheet loaded.");
        console_->setPlainText(
            "ChemSim Studio desktop shell\n"
            "Aspen/DWSIM-inspired workspace scaffold\n\n"
            "Click 'Load Example' to open the recycle flowsheet.");
        return;
    }

    appendGuiLog("Refresh: begin stream tree");
    for (const auto& name : flowsheet_->streamNames()) {
        const auto& stream = flowsheet_->getStream(name);
        auto* item = new QTreeWidgetItem(streamTree_);
        item->setText(0, QString::fromStdString(name));
        item->setText(1, QString::number(stream.totalFlow, 'f', 3));
        item->setText(2, QString::fromStdString(
            stream.phase == chemsim::Phase::VAPOR ? "VAPOR" :
            stream.phase == chemsim::Phase::LIQUID ? "LIQUID" :
            stream.phase == chemsim::Phase::MIXED ? "MIXED" : "UNKNOWN"));
    }

    appendGuiLog("Refresh: summary");
    inspector_->setPlainText(QString::fromStdString(flowsheet_->summary()));
    console_->setPlainText(
        "ChemSim Studio desktop shell\n"
        "Design direction: Aspen/DWSIM-inspired process workspace\n"
        "Loaded example: simple recycle loop\n\n"
        "Use Solve to run the recycle loop.\n"
        "Use Export Results to write the solved JSON report.");
    appendGuiLog("Refresh: done");
}
