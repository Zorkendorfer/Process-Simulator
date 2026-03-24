#include "FlowsheetCanvas.hpp"
#include <QBrush>
#include <QFont>
#include <QGraphicsItemGroup>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QPainter>
#include <QPen>

namespace {

QGraphicsItemGroup* addUnit(QGraphicsScene* scene,
                            const QRectF& rect,
                            const QString& title,
                            const QColor& fill) {
    auto* body = scene->addRect(rect, QPen(QColor("#4c5c78"), 1.0), QBrush(fill));
    body->setZValue(1.0);

    auto* label = scene->addSimpleText(title, QFont("Segoe UI Semibold", 10));
    label->setBrush(QBrush(QColor("#f5f7fb")));
    const QRectF text_rect = label->boundingRect();
    label->setPos(rect.center().x() - text_rect.width() / 2.0,
                  rect.center().y() - text_rect.height() / 2.0);
    label->setZValue(2.0);

    auto* group = new QGraphicsItemGroup();
    scene->addItem(group);
    group->addToGroup(body);
    group->addToGroup(label);
    return group;
}

void addStreamLine(QGraphicsScene* scene,
                   const QPointF& start,
                   const QPointF& end,
                   const QString& tag) {
    scene->addLine(QLineF(start, end), QPen(QColor("#93b7ff"), 2.0));
    auto* text = scene->addSimpleText(tag, QFont("Consolas", 8));
    text->setBrush(QBrush(QColor("#9fb4d9")));
    text->setPos((start.x() + end.x()) * 0.5 - 18.0, (start.y() + end.y()) * 0.5 - 18.0);
}

} // namespace

FlowsheetCanvas::FlowsheetCanvas(QWidget* parent)
    : QGraphicsView(parent),
      scene_(new QGraphicsScene(this)) {
    setScene(scene_);
    setRenderHint(QPainter::Antialiasing, true);
    setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    setBackgroundBrush(QColor("#111827"));
    setFrameShape(QFrame::NoFrame);
    scene_->setSceneRect(0.0, 0.0, 1600.0, 1000.0);
}

void FlowsheetCanvas::populateDemoDiagram() {
    scene_->clear();

    addUnit(scene_, QRectF(250.0, 280.0, 140.0, 72.0), "MIX-101", QColor("#28405d"));
    addUnit(scene_, QRectF(520.0, 280.0, 140.0, 72.0), "FLASH-101", QColor("#34596a"));
    addUnit(scene_, QRectF(840.0, 220.0, 140.0, 72.0), "PUMP-101", QColor("#3d4c7a"));
    addUnit(scene_, QRectF(840.0, 420.0, 140.0, 72.0), "SPLIT-101", QColor("#5a4469"));

    addStreamLine(scene_, QPointF(110.0, 316.0), QPointF(250.0, 316.0), "FEED");
    addStreamLine(scene_, QPointF(390.0, 316.0), QPointF(520.0, 316.0), "S1");
    addStreamLine(scene_, QPointF(660.0, 280.0), QPointF(840.0, 256.0), "VAP");
    addStreamLine(scene_, QPointF(660.0, 352.0), QPointF(840.0, 456.0), "LIQ");
    addStreamLine(scene_, QPointF(980.0, 456.0), QPointF(1110.0, 456.0), "PRODUCT");
    addStreamLine(scene_, QPointF(980.0, 256.0), QPointF(980.0, 150.0), "HP");
}

void FlowsheetCanvas::drawBackground(QPainter* painter, const QRectF& rect) {
    Q_UNUSED(rect);

    painter->fillRect(sceneRect(), QColor("#0f172a"));

    const qreal grid = 28.0;
    QPen minor(QColor("#172133"));
    QPen major(QColor("#1e293b"));

    for (qreal x = sceneRect().left(); x <= sceneRect().right(); x += grid) {
        painter->setPen((static_cast<int>(x) % static_cast<int>(grid * 4) == 0) ? major : minor);
        painter->drawLine(QLineF(x, sceneRect().top(), x, sceneRect().bottom()));
    }

    for (qreal y = sceneRect().top(); y <= sceneRect().bottom(); y += grid) {
        painter->setPen((static_cast<int>(y) % static_cast<int>(grid * 4) == 0) ? major : minor);
        painter->drawLine(QLineF(sceneRect().left(), y, sceneRect().right(), y));
    }
}
