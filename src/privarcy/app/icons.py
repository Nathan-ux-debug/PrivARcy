"""Small stroke icons drawn with QPainter - no image assets, crisp at any DPI."""

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _p(painter, pts, close=False):
    path = QPainterPath(QPointF(*pts[0]))
    for pt in pts[1:]:
        path.lineTo(QPointF(*pt))
    if close:
        path.closeSubpath()
    painter.drawPath(path)


def _draw(name, p: QPainter):
    # all icons are drawn on a 24x24 grid
    if name == "dashboard":
        for x, y, w, h in ((3.5, 3.5, 7, 8), (13.5, 3.5, 7, 5), (13.5, 11.5, 7, 9), (3.5, 14.5, 7, 6)):
            p.drawRoundedRect(QRectF(x, y, w, h), 1.6, 1.6)
    elif name == "video":
        p.drawRoundedRect(QRectF(3, 6, 13, 12), 2, 2)
        _p(p, [(16, 10.5), (21, 7.5), (21, 16.5), (16, 13.5)], True)
    elif name == "camera":
        p.drawRoundedRect(QRectF(3, 7, 18, 12), 2.5, 2.5)
        p.drawEllipse(QPointF(12, 13), 3.4, 3.4)
        _p(p, [(8.5, 7), (10, 4.5), (14, 4.5), (15.5, 7)])
    elif name == "review":
        p.drawRoundedRect(QRectF(4, 3.5, 16, 17), 2.5, 2.5)
        _p(p, [(8, 12.5), (11, 15.5), (16.5, 9)])
    elif name == "face":
        p.drawEllipse(QPointF(12, 8.5), 3.6, 3.6)
        path = QPainterPath(QPointF(4.5, 20))
        path.cubicTo(5, 14.5, 19, 14.5, 19.5, 20)
        p.drawPath(path)
    elif name == "log":
        for y in (6.5, 12, 17.5):
            p.drawLine(QPointF(9, y), QPointF(20, y))
            p.drawPoint(QPointF(4.5, y))
            p.drawEllipse(QPointF(4.8, y), 0.6, 0.6)
    elif name == "history":
        p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
        _p(p, [(12, 7.5), (12, 12), (15.2, 14)])
    elif name == "settings":
        for y, cx in ((7, 9), (12, 15.5), (17, 8)):
            p.drawLine(QPointF(4, y), QPointF(cx - 2.8, y))
            p.drawLine(QPointF(cx + 2.8, y), QPointF(20, y))
            p.drawEllipse(QPointF(cx, y), 2.2, 2.2)
    elif name == "shield":
        path = QPainterPath(QPointF(12, 3))
        path.lineTo(19.5, 6)
        path.cubicTo(19.5, 12.5, 17, 18, 12, 21)
        path.cubicTo(7, 18, 4.5, 12.5, 4.5, 6)
        path.closeSubpath()
        p.drawPath(path)
    elif name == "upload":
        _p(p, [(12, 16), (12, 4.5)])
        _p(p, [(7.5, 9), (12, 4.5), (16.5, 9)])
        _p(p, [(4.5, 16), (4.5, 20), (19.5, 20), (19.5, 16)])
    elif name == "folder":
        _p(p, [(3.5, 7), (3.5, 19), (20.5, 19), (20.5, 8.5), (11.5, 8.5), (9.5, 5.5), (3.5, 5.5)], True)
    elif name == "check":
        _p(p, [(5, 12.5), (10, 17.5), (19, 7)])
    elif name == "x":
        _p(p, [(6, 6), (18, 18)])
        _p(p, [(18, 6), (6, 18)])
    elif name == "play":
        _p(p, [(8, 5), (19, 12), (8, 19)], True)
    elif name == "stop":
        p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 1.5, 1.5)
    elif name == "refresh":
        p.drawArc(QRectF(5, 5, 14, 14), 40 * 16, 270 * 16)
        _p(p, [(16.5, 3.5), (17.5, 7.5), (13.5, 8.2)])
    elif name == "arrow-right":
        _p(p, [(4.5, 12), (19.5, 12)])
        _p(p, [(14, 6.5), (19.5, 12), (14, 17.5)])
    elif name == "bell":
        p.drawArc(QRectF(6, 4.5, 12, 12), 0, 180 * 16)
        _p(p, [(6, 10.5), (6, 16.5)]); _p(p, [(18, 10.5), (18, 16.5)]); _p(p, [(4.5, 16.5), (19.5, 16.5)])
        p.drawArc(QRectF(10, 17.5, 4, 3), 180 * 16, 180 * 16)
    elif name == "user":
        _draw("face", p)
    elif name == "lock":
        p.drawRoundedRect(QRectF(5.5, 11, 13, 9.5), 2, 2)
        p.drawArc(QRectF(8, 4, 8, 14), 0, 180 * 16)
    elif name == "target":
        p.drawEllipse(QPointF(12, 12), 8.5, 8.5); p.drawEllipse(QPointF(12, 12), 3.8, 3.8)
    elif name == "cpu":
        p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2, 2)
        for v in (9.5, 14.5):
            p.drawLine(QPointF(v, 3), QPointF(v, 6.5)); p.drawLine(QPointF(v, 17.5), QPointF(v, 21))
            p.drawLine(QPointF(3, v), QPointF(6.5, v)); p.drawLine(QPointF(17.5, v), QPointF(21, v))
    elif name == "doc":
        _p(p, [(6, 3.5), (14, 3.5), (18.5, 8), (18.5, 20.5), (6, 20.5)], True)
        _p(p, [(9, 12), (15.5, 12)]); _p(p, [(9, 16), (15.5, 16)])
    elif name == "bug":
        p.drawEllipse(QPointF(12, 13.5), 4.2, 5.8)
        for y in (10.5, 14, 17.5):
            p.drawLine(QPointF(3.5, y - 1), QPointF(7.8, y)); p.drawLine(QPointF(20.5, y - 1), QPointF(16.2, y))
        _p(p, [(9.5, 5), (11, 8)]); _p(p, [(14.5, 5), (13, 8)])
    elif name == "pen":
        _p(p, [(4.5, 19.5), (5.5, 15), (16, 4.5), (19.5, 8), (9, 18.5)], True)
    elif name == "plus":
        _p(p, [(12, 5), (12, 19)]); _p(p, [(5, 12), (19, 12)])
    elif name == "warn":
        _p(p, [(12, 4), (21, 19.5), (3, 19.5)], True); _p(p, [(12, 10), (12, 14.5)]); p.drawPoint(QPointF(12, 17))
    elif name == "chevron-left":
        _p(p, [(15, 5.5), (8.5, 12), (15, 18.5)])
    elif name == "chevron-right":
        _p(p, [(9, 5.5), (15.5, 12), (9, 18.5)])
    elif name == "trash":
        _p(p, [(5, 7), (19, 7)]); _p(p, [(9.5, 7), (9.5, 4), (14.5, 4), (14.5, 7)]); _p(p, [(7, 7), (8, 20.5), (16, 20.5), (17, 7)])
    elif name == "eye":
        path = QPainterPath(QPointF(3, 12)); path.cubicTo(7, 5.5, 17, 5.5, 21, 12); path.cubicTo(17, 18.5, 7, 18.5, 3, 12)
        p.drawPath(path); p.drawEllipse(QPointF(12, 12), 3, 3)
    elif name == "phone":
        p.drawRoundedRect(QRectF(7, 3, 10, 18), 2.2, 2.2); p.drawPoint(QPointF(12, 18))
    elif name == "network":
        p.drawEllipse(QPointF(12, 12), 8.5, 8.5); p.drawEllipse(QPointF(12, 12), 3.6, 8.5); _p(p, [(3.5, 12), (20.5, 12)])
    elif name == "wand":
        _p(p, [(5, 19), (15, 9)]); _p(p, [(15, 4), (15, 8)]); _p(p, [(13, 6), (17, 6)]); _p(p, [(19.5, 11), (19.5, 14)]); _p(p, [(18, 12.5), (21, 12.5)])
    elif name == "region":
        pen = p.pen(); pen.setStyle(Qt.DashLine); p.setPen(pen)
        p.drawRoundedRect(QRectF(4, 5, 16, 14), 1.5, 1.5)
    elif name == "sparkle":
        _p(p, [(12, 3.5), (13.8, 10.2), (20.5, 12), (13.8, 13.8), (12, 20.5), (10.2, 13.8), (3.5, 12), (10.2, 10.2)], True)
    else:
        p.drawEllipse(QPointF(12, 12), 7, 7)


def make_icon(name: str, color: str = "#9BA1AC", size: int = 18, stroke: float = 1.7) -> QIcon:
    scale = 2
    pm = QPixmap(size * scale, size * scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.scale(size * scale / 24.0, size * scale / 24.0)
    pen = QPen(QColor(color), stroke)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    _draw(name, p)
    p.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


def make_pixmap(name: str, color: str, size: int = 18):
    return make_icon(name, color, size).pixmap(size, size)
