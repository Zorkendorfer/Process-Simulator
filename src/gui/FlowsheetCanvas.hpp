#pragma once
#include <functional>
#include <QGraphicsView>

class QGraphicsScene;

class FlowsheetCanvas : public QGraphicsView {
public:
    explicit FlowsheetCanvas(QWidget* parent = nullptr);

    void populateDemoDiagram();
    void setSelectionChangedCallback(std::function<void(const QString&)> callback);

protected:
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void handleSelectionChanged();

    QGraphicsScene* scene_;
    std::function<void(const QString&)> selection_callback_;
};
