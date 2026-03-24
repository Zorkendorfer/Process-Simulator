#pragma once
#include <QGraphicsView>

class QGraphicsScene;

class FlowsheetCanvas : public QGraphicsView {
public:
    explicit FlowsheetCanvas(QWidget* parent = nullptr);

    void populateDemoDiagram();

protected:
    void drawBackground(QPainter* painter, const QRectF& rect) override;

private:
    QGraphicsScene* scene_;
};
