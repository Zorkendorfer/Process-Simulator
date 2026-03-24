#include "FlowsheetCanvas.hpp"
#include <QBrush>
#include <QFont>
#include <QGraphicsDropShadowEffect>
#include <QGraphicsObject>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QWheelEvent>

namespace {

class StreamLineItem;

class UnitBlockItem : public QGraphicsObject {
public:
    UnitBlockItem(QString tag, QString title, QColor fill, QGraphicsItem* parent = nullptr)
        : QGraphicsObject(parent),
          tag_(std::move(tag)),
          title_(std::move(title)),
          fill_(std::move(fill)) {
        setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsGeometryChanges);
        setCacheMode(DeviceCoordinateCache);

        auto* shadow = new QGraphicsDropShadowEffect();
        shadow->setBlurRadius(22.0);
        shadow->setOffset(0.0, 8.0);
        shadow->setColor(QColor(6, 10, 20, 130));
        setGraphicsEffect(shadow);
    }

    QRectF boundingRect() const override {
        return QRectF(0.0, 0.0, 160.0, 84.0);
    }

    void addStream(StreamLineItem* stream) {
        streams_.push_back(stream);
    }

    QPointF anchorLeft() const {
        return mapToScene(QPointF(0.0, boundingRect().center().y()));
    }

    QPointF anchorRight() const {
        return mapToScene(QPointF(boundingRect().right(), boundingRect().center().y()));
    }

    QString inspectorText() const {
        return QString(
            "Unit Operation\n"
            "--------------\n"
            "Tag: %1\n"
            "Type: %2\n"
            "Position: (%3, %4)\n\n"
            "Drag this block to rearrange the flowsheet canvas.")
            .arg(tag_)
            .arg(title_)
            .arg(pos().x(), 0, 'f', 1)
            .arg(pos().y(), 0, 'f', 1);
    }

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override {
        const QRectF rect = boundingRect();
        QLinearGradient grad(rect.topLeft(), rect.bottomRight());
        grad.setColorAt(0.0, fill_.lighter(120));
        grad.setColorAt(1.0, fill_.darker(135));

        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->setPen(QPen(isSelected() ? QColor("#b8d4ff") : QColor("#4c5c78"),
                             isSelected() ? 2.2 : 1.2));
        painter->setBrush(grad);
        painter->drawRoundedRect(rect, 12.0, 12.0);

        painter->setPen(QColor("#7ea3d7"));
        painter->setFont(QFont("Segoe UI", 8, QFont::DemiBold));
        painter->drawText(QRectF(14.0, 10.0, rect.width() - 28.0, 18.0),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          tag_);

        painter->setPen(QColor("#eef4ff"));
        painter->setFont(QFont("Segoe UI Semibold", 10));
        painter->drawText(QRectF(14.0, 28.0, rect.width() - 28.0, 28.0),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          title_);

        painter->setPen(QColor("#9fb4d9"));
        painter->setFont(QFont("Consolas", 8));
        painter->drawText(QRectF(14.0, 56.0, rect.width() - 28.0, 18.0),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          "Interactive block");
    }

private:
    QString tag_;
    QString title_;
    QColor fill_;
    std::vector<StreamLineItem*> streams_;
};

class StreamLineItem : public QGraphicsPathItem {
public:
    StreamLineItem(UnitBlockItem* from,
                   UnitBlockItem* to,
                   QString tag,
                   QGraphicsItem* parent = nullptr)
        : QGraphicsPathItem(parent),
          from_(from),
          to_(to),
          tag_(std::move(tag)) {
        setFlag(ItemIsSelectable, true);
        setZValue(0.5);
        updateGeometry();
    }

    void updateGeometry() {
        if (!from_ || !to_) {
            return;
        }

        const QPointF start = from_->anchorRight();
        const QPointF end = to_->anchorLeft();
        const qreal dx = std::max<qreal>(90.0, (end.x() - start.x()) * 0.45);

        QPainterPath p(start);
        p.cubicTo(start + QPointF(dx, 0.0), end - QPointF(dx, 0.0), end);
        setPath(p);

        const QColor stroke = isSelected() ? QColor("#d7e6ff") : QColor("#93b7ff");
        setPen(QPen(stroke, isSelected() ? 3.0 : 2.2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    }

    QString inspectorText() const {
        return QString(
            "Material Stream\n"
            "---------------\n"
            "Tag: %1\n"
            "From: %2\n"
            "To: %3\n\n"
            "Streams stay attached while unit blocks move.")
            .arg(tag_)
            .arg(from_ ? from_->data(0).toString() : QString("Feed"))
            .arg(to_ ? to_->data(0).toString() : QString("Product"));
    }

    QString tag() const { return tag_; }

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override {
        if (change == ItemSelectedHasChanged) {
            updateGeometry();
        }
        return QGraphicsPathItem::itemChange(change, value);
    }

private:
    UnitBlockItem* from_;
    UnitBlockItem* to_;
    QString tag_;
};

QVariant UnitBlockItem::itemChange(GraphicsItemChange change, const QVariant& value) {
    if (change == ItemPositionHasChanged) {
        for (auto* stream : streams_) {
            if (stream) {
                stream->updateGeometry();
            }
        }
    }
    if (change == ItemSelectedHasChanged) {
        update();
    }
    return QGraphicsObject::itemChange(change, value);
}

QGraphicsSimpleTextItem* addFreeLabel(QGraphicsScene* scene,
                                      const QPointF& position,
                                      const QString& text) {
    auto* label = scene->addSimpleText(text, QFont("Consolas", 8));
    label->setBrush(QBrush(QColor("#9fb4d9")));
    label->setPos(position);
    label->setZValue(1.5);
    return label;
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
    setDragMode(QGraphicsView::RubberBandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    scene_->setSceneRect(0.0, 0.0, 1800.0, 1100.0);

    QObject::connect(scene_, &QGraphicsScene::selectionChanged, this,
                     [this]() { handleSelectionChanged(); });
}

void FlowsheetCanvas::setSelectionChangedCallback(std::function<void(const QString&)> callback) {
    selection_callback_ = std::move(callback);
}

void FlowsheetCanvas::populateDemoDiagram() {
    scene_->clear();

    auto* mix = new UnitBlockItem("MIX-101", "Mixer", QColor("#28405d"));
    auto* flash = new UnitBlockItem("FLASH-101", "Flash Drum", QColor("#34596a"));
    auto* pump = new UnitBlockItem("PUMP-101", "Pump", QColor("#3d4c7a"));
    auto* split = new UnitBlockItem("SPLIT-101", "Splitter", QColor("#5a4469"));

    mix->setData(0, "MIX-101");
    flash->setData(0, "FLASH-101");
    pump->setData(0, "PUMP-101");
    split->setData(0, "SPLIT-101");

    scene_->addItem(mix);
    scene_->addItem(flash);
    scene_->addItem(pump);
    scene_->addItem(split);

    mix->setPos(260.0, 310.0);
    flash->setPos(560.0, 310.0);
    pump->setPos(910.0, 230.0);
    split->setPos(910.0, 450.0);

    auto* s1 = new StreamLineItem(mix, flash, "S1");
    auto* vap = new StreamLineItem(flash, pump, "VAP");
    auto* liq = new StreamLineItem(flash, split, "LIQ");

    scene_->addItem(s1);
    scene_->addItem(vap);
    scene_->addItem(liq);

    mix->addStream(s1);
    flash->addStream(s1);
    flash->addStream(vap);
    flash->addStream(liq);
    pump->addStream(vap);
    split->addStream(liq);

    scene_->addLine(QLineF(120.0, 352.0, 260.0, 352.0),
                    QPen(QColor("#93b7ff"), 2.2, Qt::SolidLine, Qt::RoundCap));
    addFreeLabel(scene_, QPointF(150.0, 326.0), "FEED");

    scene_->addLine(QLineF(1070.0, 272.0, 1210.0, 272.0),
                    QPen(QColor("#93b7ff"), 2.2, Qt::SolidLine, Qt::RoundCap));
    addFreeLabel(scene_, QPointF(1100.0, 246.0), "HP");

    scene_->addLine(QLineF(1070.0, 492.0, 1220.0, 492.0),
                    QPen(QColor("#93b7ff"), 2.2, Qt::SolidLine, Qt::RoundCap));
    addFreeLabel(scene_, QPointF(1110.0, 466.0), "PRODUCT");

    auto* recyclePath = new QGraphicsPathItem();
    QPainterPath recycle(QPointF(1070.0, 492.0));
    recycle.cubicTo(QPointF(1220.0, 580.0), QPointF(190.0, 580.0), QPointF(190.0, 352.0));
    recyclePath->setPath(recycle);
    recyclePath->setPen(QPen(QColor("#5fb0ff"), 2.0, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin));
    recyclePath->setFlag(QGraphicsItem::ItemIsSelectable, true);
    scene_->addItem(recyclePath);
    addFreeLabel(scene_, QPointF(610.0, 598.0), "RECYCLE LOOP");

    if (selection_callback_) {
        selection_callback_(
            "Workspace\n"
            "---------\n"
            "Drag equipment blocks to reorganize the layout.\n"
            "Click a unit or stream to inspect it.\n"
            "Use the mouse wheel to zoom the flowsheet.");
    }
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

void FlowsheetCanvas::wheelEvent(QWheelEvent* event) {
    constexpr qreal zoom_in = 1.12;
    constexpr qreal zoom_out = 1.0 / zoom_in;

    if (event->angleDelta().y() > 0) {
        scale(zoom_in, zoom_in);
    } else {
        scale(zoom_out, zoom_out);
    }
}

void FlowsheetCanvas::handleSelectionChanged() {
    if (!selection_callback_) {
        return;
    }

    const auto items = scene_->selectedItems();
    if (items.empty()) {
        selection_callback_(
            "Workspace\n"
            "---------\n"
            "Drag equipment blocks to reorganize the layout.\n"
            "Click a unit or stream to inspect it.\n"
            "Use the mouse wheel to zoom the flowsheet.");
        return;
    }

    if (auto* unit = dynamic_cast<UnitBlockItem*>(items.front())) {
        selection_callback_(unit->inspectorText());
        return;
    }

    if (auto* stream = dynamic_cast<StreamLineItem*>(items.front())) {
        selection_callback_(stream->inspectorText());
        return;
    }

    selection_callback_("Canvas item selected.");
}
