"""Control panel: object toggles, styles, effects, mission, live stats."""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QCompleter, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLayout, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from .config import PROMPT_COLORS, Config
from .detector import CUSTOM_BASE
from .vocab import Vocabulary

MAX_PROMPTS = 8
NOT_SEEN_S = 6.0  # after this long without a sighting, a typed word is flagged as "not seen yet"
TRY_WORDS = ("watch", "glasses", "hat", "bracelet", "lamp")  # words YOLOE-26 finds reliably

QSS = """
#Sidebar { background: #0b0e18; border-left: 1px solid rgba(255,255,255,0.06); }
QScrollArea, #SidebarInner { background: transparent; border: none; }
QLabel { color: #d7deef; }
QLabel#Logo { font-size: 22px; font-weight: 900; letter-spacing: 3px; color: #ffffff; }
QLabel#Tagline { font-size: 11px; color: #7c87a6; }
QLabel#Section { font-size: 10px; font-weight: 800; letter-spacing: 2px; color: #6f7a99; padding-top: 6px; }
QLabel#Value { font-size: 12px; font-weight: 700; color: #3DE8FF; }
QLabel#Stat { font-size: 11px; color: #9aa4c0; }
QLabel#StatBig { font-size: 18px; font-weight: 800; color: #ffffff; }
QLabel#Hint { font-size: 10px; color: #586384; }
QLabel#Badge { font-size: 11px; font-weight: 800; border-radius: 9px; padding: 1px 8px;
               min-width: 12px; background: rgba(255,255,255,0.05); color: #586384; }

QPushButton { color: #c9d1e4; background: rgba(255,255,255,0.05);
              border: 1px solid rgba(255,255,255,0.09); border-radius: 10px;
              padding: 7px 10px; font-size: 12px; font-weight: 700; }
QPushButton:hover { border-color: rgba(255,255,255,0.25); background: rgba(255,255,255,0.08); }
QPushButton:checked { color: #ffffff; border: 1px solid rgba(61,232,255,0.75);
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(61,232,255,0.28), stop:1 rgba(255,61,154,0.24)); }
QPushButton#Mini { padding: 2px 9px; font-size: 10px; border-radius: 8px; }
QPushButton#Mission { font-size: 14px; font-weight: 900; padding: 11px; border-radius: 12px; color: #0a0c14;
    border: none; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3DE8FF, stop:1 #FF3D9A); }
QPushButton#Mission:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7ff0ff, stop:1 #ff7ab8); }
QPushButton#Mission:checked { color: #ffffff; border: 1px solid #FF4D5E; background: rgba(255,77,94,0.18); }

QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: rgba(255,255,255,0.10); }
QSlider::sub-page:horizontal { border-radius: 2px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3DE8FF, stop:1 #FF3D9A); }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: #ffffff; }
QLineEdit { color: #ffffff; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px; padding: 8px 10px; font-size: 12px; selection-background-color: #3DE8FF; }
QLineEdit:focus { border: 1px solid rgba(124,255,203,0.8); background: rgba(124,255,203,0.06); }
QLabel#Try { font-size: 10px; color: #586384; }
QToolTip { color: #ffffff; background: #121726; border: 1px solid rgba(61,232,255,0.6);
           border-radius: 6px; padding: 4px 8px; font-size: 12px; }
QScrollBar:vertical { width: 6px; background: transparent; }
QScrollBar::handle:vertical { background: rgba(255,255,255,0.12); border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def section(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("Section")
    return lbl


class ObjectRow(QWidget):
    toggled = Signal()

    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.color = obj.color
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 1, 2, 1)
        self.box = QCheckBox(f"{obj.icon}   {obj.label}")
        self.box.setChecked(obj.enabled)
        self.box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.box.setStyleSheet(f"""
            QCheckBox {{ color: #d7deef; font-size: 13px; font-weight: 600; spacing: 10px; padding: 4px 0; }}
            QCheckBox:unchecked {{ color: #6f7a99; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 5px;
                                    border: 2px solid {obj.color}; background: transparent; }}
            QCheckBox::indicator:checked {{ background: {obj.color}; }}
        """)
        self.box.toggled.connect(lambda _=None: self.toggled.emit())
        self.box.setToolTip(f"Tick to look for: {obj.label}")
        self.badge = QLabel("0")
        self.badge.setToolTip("How many are on screen now")
        self.badge.setObjectName("Badge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.box, 1)
        lay.addWidget(self.badge)
        self._count = -1
        self.set_count(0)

    @property
    def checked(self) -> bool:
        return self.box.isChecked()

    def set_count(self, n: int):
        if n == self._count:
            return
        self._count = n
        self.badge.setText(str(n))
        if n > 0:
            self.badge.setStyleSheet(f"background: {self.color}; color: #0a0c14;")
        else:
            self.badge.setStyleSheet("")


POPUP_QSS = """
QListView { background: #121726; color: #d7deef; border: 1px solid rgba(124,255,203,0.5);
            border-radius: 8px; padding: 4px; font-size: 12px; outline: 0; }
QListView::item { padding: 4px 8px; border-radius: 5px; }
QListView::item:selected { background: rgba(124,255,203,0.22); color: #ffffff; }
"""


class FlowLayout(QLayout):
    """Lays widgets out left to right and wraps to a new line when the panel is too narrow."""

    def __init__(self, parent=None, spacing: int = 4):
        super().__init__(parent)
        self._items = []
        self._gap = spacing
        self.setContentsMargins(2, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect, move: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for item in self._items:
            w = item.sizeHint()
            if x + w.width() > r.right() + 1 and line_h > 0:
                x, y, line_h = r.x(), y + line_h + self._gap, 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), w))
            x += w.width() + self._gap
            line_h = max(line_h, w.height())
        return y + line_h - rect.y() + m.bottom()


class PanelContent(QWidget):
    """Scrollable panel body: asks for its full height, so a short window scrolls instead of squashing rows."""

    def minimumSizeHint(self):
        lay = self.layout()
        if lay is None:
            return super().minimumSizeHint()
        h = lay.totalHeightForWidth(self.width()) if lay.hasHeightForWidth() else -1
        return QSize(lay.totalMinimumSize().width(), max(h, lay.totalSizeHint().height()))


class TermCompleter(QCompleter):
    """Autocompletes the word after the last comma, so several words can be typed in one box."""

    def __init__(self, terms, edit: QLineEdit):
        super().__init__(terms, edit)
        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterMode(Qt.MatchFlag.MatchStartsWith)
        self.setMaxVisibleItems(8)
        self.popup().setStyleSheet(POPUP_QSS)

    def splitPath(self, path):
        return [path.split(",")[-1].strip()]

    def pathFromIndex(self, index):
        item = super().pathFromIndex(index)
        head = self.widget().text().rsplit(",", 1)
        return f"{head[0]}, {item}" if len(head) > 1 else item


class PromptRow(QWidget):
    """One typed word: its box color, live count, and whether YOLOE has found it yet."""

    def __init__(self, label: str, color: str | None):
        super().__init__()
        self.color = color
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 2, 0)
        lay.setSpacing(8)
        dot = QLabel("●" if color else "✗")
        dot.setStyleSheet(f"color: {color or '#FF6B7A'}; font-size: 13px;")
        name = QLabel(label)
        name.setStyleSheet("font-size: 12px; font-weight: 600;" if color else
                           "font-size: 12px; color: #FF6B7A; text-decoration: line-through;")
        self.status = QLabel("")
        self.status.setStyleSheet("font-size: 10px; color: #6f7a99;")
        lay.addWidget(dot)
        lay.addWidget(name, 1)
        lay.addWidget(self.status)
        self.badge = None
        if color:
            self.badge = QLabel("0")
            self.badge.setObjectName("Badge")
            self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(self.badge)
            self.setToolTip("Box color for this word · how many are on screen now")
        else:
            self.set_status("unknown word", "#FF6B7A")
            self.setToolTip("Not a known object name, so it is not searched")
        self.count = 0
        self.seen = False

    def set_status(self, text: str, color: str = "#6f7a99"):
        self.status.setText(text)
        self.status.setStyleSheet(f"font-size: 10px; color: {color};")

    def set_count(self, n: int):
        if self.badge is None or n == self.count:
            return
        self.count = n
        self.badge.setText(str(n))
        self.badge.setStyleSheet(f"background: {self.color}; color: #0a0c14;" if n else "")


class Sparkline(QWidget):
    def __init__(self):
        super().__init__()
        self.values: list[float] = []
        self.setFixedHeight(46)

    def set_values(self, values):
        self.values = list(values)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 4, -1, -2)
        p.setPen(QPen(QColor(255, 255, 255, 22), 1, Qt.PenStyle.DashLine))
        top = max(65.0, max(self.values, default=0) * 1.1)
        y30 = r.bottom() - r.height() * 30 / top
        p.drawLine(QPointF(r.left(), y30), QPointF(r.right(), y30))
        if len(self.values) < 2:
            return
        n = len(self.values)
        pts = [QPointF(r.left() + r.width() * i / (n - 1), r.bottom() - r.height() * min(v, top) / top)
               for i, v in enumerate(self.values)]
        line = QPainterPath(pts[0])
        for pt in pts[1:]:
            line.lineTo(pt)
        area = QPainterPath(line)
        area.lineTo(pts[-1].x(), r.bottom())
        area.lineTo(pts[0].x(), r.bottom())
        g = QLinearGradient(0, r.top(), 0, r.bottom())
        g.setColorAt(0, QColor(61, 232, 255, 90))
        g.setColorAt(1, QColor(61, 232, 255, 0))
        p.fillPath(area, g)
        p.setPen(QPen(QColor("#3DE8FF"), 2))
        p.drawPath(line)


class Sidebar(QFrame):
    classes_changed = Signal(list)
    confidence_changed = Signal(float)
    style_changed = Signal(str)
    effect_toggled = Signal(str, bool)
    prompts_changed = Signal(list)
    mission_toggled = Signal(bool)
    mission_skip = Signal()
    model_changed = Signal(str)

    MODEL_TIPS = {"light": "Fastest, a bit less accurate",
                  "medium": "Balanced speed and accuracy",
                  "heavy": "Most accurate, slower with AI layers on"}
    snapshot_requested = Signal()
    camera_requested = Signal()

    LAYERS = (
        ("masks", "🧬  Masks", "Outline the exact shape of each object · key G"),
        ("pose", "🕺  Pose", "Draw a stick-figure skeleton on people · key K"),
        ("depth", "📏  Depth", "Show how far things are (colors + meters) · key D"),
        ("portrait", "🌫️  Portrait", "Blur the background behind the closest object · key B"),
    )
    EFFECTS = (
        ("trails", "✨  Trails", True, "Draw a short path behind moving objects · key T"),
        ("spotlight", "🔦  Spotlight", False, "Darken everything except the found objects · key L"),
        ("heatmap", "🔥  Heatmap", False, "Glow where objects show up most often · key H"),
        ("privacy", "🙈  Privacy", False, "Pixelate people's faces · key P"),
        ("voice", "🔊  Voice", False, "Say each new object's name out loud · key V"),
        ("mirror", "🪞  Mirror", True, "Flip the picture left-right, like a mirror"),
    )

    def __init__(self, cfg: Config):
        super().__init__()
        self.setObjectName("Sidebar")
        self.setStyleSheet(QSS)
        self.setFixedWidth(300)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        inner = PanelContent()
        inner.setObjectName("SidebarInner")
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(18, 18, 16, 16)
        lay.setSpacing(6)

        logo = QLabel("NEON<span style='color:#3DE8FF'>SIGHT</span>")
        logo.setObjectName("Logo")
        lay.addWidget(logo)
        self.tagline = tag = QLabel(f"Real-time {cfg.model_label} vision · Apple GPU")
        tag.setObjectName("Tagline")
        lay.addWidget(tag)
        lay.addSpacing(8)

        # --- objects
        head = QHBoxLayout()
        head.addWidget(section("Detect"))
        head.addStretch(1)
        for text, state, tip in (("All", True, "Tick every object"), ("None", False, "Untick every object")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setObjectName("Mini")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, s=state: self.set_all(s))
            head.addWidget(b)
        lay.addLayout(head)
        self.rows: list[ObjectRow] = []
        for obj in cfg.objects:
            row = ObjectRow(obj)
            row.toggled.connect(self._emit_classes)
            self.rows.append(row)
            lay.addWidget(row)

        # --- open vocabulary
        lay.addSpacing(6)
        lay.addWidget(section("Ask for anything"))
        self.vocab = Vocabulary()
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("type e.g.  watch, glasses, hat  ⏎")
        self.prompt.setClearButtonEnabled(True)
        self.prompt.setToolTip("Type any object names, separated by commas, then press Enter")
        self.prompt.setCompleter(TermCompleter(self.vocab.terms, self.prompt))
        self.prompt.returnPressed.connect(self._emit_prompts)
        self.prompt.textChanged.connect(lambda t: (not t.strip()) and self.prompt_rows and self._emit_prompts())
        lay.addWidget(self.prompt)
        self.try_row = QWidget()
        tlay = FlowLayout(self.try_row)
        lbl = QLabel("try")
        lbl.setObjectName("Try")
        tlay.addWidget(lbl)
        for word in TRY_WORDS:
            b = QPushButton(word)
            b.setObjectName("Mini")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"Look for: {word}")
            b.clicked.connect(lambda _=False, w=word: self._add_word(w))
            tlay.addWidget(b)
        lay.addWidget(self.try_row)
        self.prompt_box = QVBoxLayout()
        self.prompt_box.setContentsMargins(0, 2, 0, 0)
        self.prompt_box.setSpacing(2)
        lay.addLayout(self.prompt_box)
        self.prompt_note = QLabel("")
        self.prompt_note.setWordWrap(True)
        self.prompt_note.hide()
        lay.addWidget(self.prompt_note)
        self.fix_box = QVBoxLayout()
        self.fix_box.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.fix_box)
        self._note_kind = ""
        self.prompt_rows: list[PromptRow] = []
        self._accepted: list[PromptRow] = []
        self._fix_buttons: list[QPushButton] = []
        self._live_since: float | None = None
        self._prompt_timer = QTimer(self)
        self._prompt_timer.timeout.connect(self._refresh_prompt_status)

        # --- AI layers
        lay.addSpacing(6)
        lay.addWidget(section("AI layers"))
        lgrid = QGridLayout()
        lgrid.setSpacing(6)
        self.layer_buttons = {}
        for i, (key, text, tip) in enumerate(self.LAYERS):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet("text-align: left; padding-left: 12px;")
            b.toggled.connect(lambda on, k=key: self.effect_toggled.emit(k, on))
            self.layer_buttons[key] = b
            lgrid.addWidget(b, i // 2, i % 2)
        lay.addLayout(lgrid)

        # --- confidence
        lay.addSpacing(6)
        ch = QHBoxLayout()
        ch.addWidget(section("Confidence"))
        ch.addStretch(1)
        self.conf_value = QLabel()
        self.conf_value.setObjectName("Value")
        ch.addWidget(self.conf_value)
        lay.addLayout(ch)
        self.conf = QSlider(Qt.Orientation.Horizontal)
        self.conf.setRange(10, 90)
        self.conf.setValue(int(cfg.confidence * 100))
        self.conf.setToolTip("How sure the AI must be to draw a box. Higher = fewer boxes, fewer mistakes")
        self.conf.valueChanged.connect(self._emit_conf)
        lay.addWidget(self.conf)
        self._emit_conf(self.conf.value())

        # --- model size
        lay.addSpacing(6)
        lay.addWidget(section("Model"))
        prow = QHBoxLayout()
        prow.setSpacing(6)
        self.model_group = QButtonGroup(self)
        self.model_buttons = {}
        for key, spec in cfg.presets.items():
            b = QPushButton(key.capitalize())
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"{self.MODEL_TIPS.get(key, 'Detector model')} ({spec['label']})")
            self.model_group.addButton(b)
            self.model_buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self.model_changed.emit(k))
            prow.addWidget(b)
        self.model_buttons[cfg.preset].setChecked(True)
        lay.addLayout(prow)

        # --- style
        lay.addSpacing(6)
        lay.addWidget(section("Annotation style"))
        srow = QHBoxLayout()
        srow.setSpacing(6)
        self.style_group = QButtonGroup(self)
        self.style_buttons = {}
        for key, text, tip in (("neon", "Neon", "Glowing neon boxes · key 1"),
                               ("glass", "Glass", "Soft frosted-glass boxes · key 2"),
                               ("target", "Target", "Spinning target rings with a pointer label · key 3")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.style_group.addButton(b)
            self.style_buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self.style_changed.emit(k))
            srow.addWidget(b)
        self.style_buttons["neon"].setChecked(True)
        lay.addLayout(srow)

        # --- effects
        lay.addSpacing(6)
        lay.addWidget(section("Effects"))
        grid = QGridLayout()
        grid.setSpacing(6)
        self.effect_buttons = {}
        for i, (key, text, default, tip) in enumerate(self.EFFECTS):
            if key == "mirror":
                default = cfg.mirror
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(default)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet("text-align: left; padding-left: 12px;")
            b.setToolTip(tip)
            b.toggled.connect(lambda on, k=key: self.effect_toggled.emit(k, on))
            self.effect_buttons[key] = b
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)

        # --- mission
        lay.addSpacing(10)
        self.mission_btn = QPushButton("🎯  START MISSION")
        self.mission_btn.setObjectName("Mission")
        self.mission_btn.setCheckable(True)
        self.mission_btn.setToolTip("Game: show the object it asks for before time runs out · key M")
        self.mission_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mission_btn.toggled.connect(self._mission_toggled)
        lay.addWidget(self.mission_btn)
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        self.mission_status = QLabel(f"{cfg.mission_rounds} rounds · {cfg.mission_seconds:g}s each")
        self.mission_status.setObjectName("Stat")
        self.mission_status.setToolTip("Mission progress: round, objects found, points")
        self.skip_btn = QPushButton("⏭  Skip")
        self.skip_btn.setObjectName("Mini")
        self.skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skip_btn.setToolTip("Don't have this object? Skip to the next one · key N")
        self.skip_btn.clicked.connect(self.mission_skip.emit)
        self.skip_btn.setEnabled(False)
        mrow.addWidget(self.mission_status, 1)
        mrow.addWidget(self.skip_btn)
        lay.addLayout(mrow)

        arow = QHBoxLayout()
        arow.setSpacing(6)
        snap = QPushButton("📸  Snapshot")
        snap.setCursor(Qt.CursorShape.PointingHandCursor)
        snap.setToolTip("Save a picture of the view to the snapshots folder · key S")
        snap.clicked.connect(self.snapshot_requested.emit)
        cam = QPushButton("🎥  Camera")
        cam.setCursor(Qt.CursorShape.PointingHandCursor)
        cam.setToolTip("Switch to the next camera · key C")
        cam.clicked.connect(self.camera_requested.emit)
        arow.addWidget(snap)
        arow.addWidget(cam)
        lay.addLayout(arow)

        # --- stats
        lay.addSpacing(8)
        lay.addWidget(section("Performance"))
        stats = QGridLayout()
        stats.setVerticalSpacing(0)
        self.stat_fps = QLabel("–")
        self.stat_ms = QLabel("–")
        self.stat_seen = QLabel("0")
        for col, (big, caption, tip) in enumerate((
                (self.stat_fps, "FPS", "Pictures shown per second"),
                (self.stat_ms, "AI ms", "Milliseconds the AI needs per picture"),
                (self.stat_seen, "seen", "Different objects found since the start"))):
            big.setObjectName("StatBig")
            big.setToolTip(tip)
            cap = QLabel(caption)
            cap.setToolTip(tip)
            cap.setObjectName("Stat")
            stats.addWidget(big, 0, col)
            stats.addWidget(cap, 1, col)
        lay.addLayout(stats)
        self.spark = Sparkline()
        self.spark.setToolTip("Speed over the last seconds · dashed line = 30 FPS")
        lay.addWidget(self.spark)

        lay.addStretch(1)
        hint = QLabel("Space pause · S snap · M mission · N skip\n"
                      "G masks · K pose · D depth · B portrait · T trails\n"
                      "L spotlight · H heatmap · P privacy · V voice\n"
                      "1/2/3 style · C camera · Tab panel · F fullscreen\n"
                      "Q quit")
        hint.setObjectName("Hint")
        hint.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay.addWidget(hint)

    # ------------------------------------------------------------ helpers
    def selected_class_ids(self) -> list[int]:
        return [r.obj.class_id for r in self.rows if r.checked]

    def set_all(self, state: bool):
        for r in self.rows:
            r.box.blockSignals(True)
            r.box.setChecked(state)
            r.box.blockSignals(False)
        self._emit_classes()

    def _emit_classes(self):
        self.classes_changed.emit(self.selected_class_ids())

    # ------------------------------------------------------- open vocabulary
    def _add_word(self, word: str):
        parts = [p.strip() for p in self.prompt.text().split(",") if p.strip()]
        if word not in parts:
            parts.append(word)
        self.prompt.setText(", ".join(parts))
        self._emit_prompts()

    def _replace_word(self, raw: str, fix: str):
        parts = [p.strip() for p in self.prompt.text().split(",") if p.strip()]
        self.prompt.setText(", ".join(fix if p == raw else p for p in parts))
        self._emit_prompts()

    def _emit_prompts(self):
        """Check each typed word against the vocabulary; only known object names reach YOLOE."""
        for w in self.prompt_rows + self._fix_buttons:
            w.hide()
            w.setParent(None)  # detach now; Qt frees it later
            w.deleteLater()
        self.prompt_rows, self._accepted, self._fix_buttons = [], [], []
        accepted: list[str] = []
        rejected: list[tuple[str, list[str]]] = []
        seen_raw = set()
        for raw in self.prompt.text().split(","):
            raw = raw.strip()
            if not raw or raw.lower() in seen_raw:
                continue
            seen_raw.add(raw.lower())
            term, suggestions = self.vocab.check(raw)
            if term is None:
                rejected.append((raw, suggestions))
            elif term not in accepted and len(accepted) < MAX_PROMPTS:
                accepted.append(term)

        for i, term in enumerate(accepted):
            row = PromptRow(term, PROMPT_COLORS[i % len(PROMPT_COLORS)])
            row.set_status("starting…")
            self._accepted.append(row)
        rows = list(self._accepted)
        for raw, _ in rejected:
            rows.append(PromptRow(raw, None))
        for row in rows:
            self.prompt_box.addWidget(row)
        self.prompt_rows = rows

        if rejected:
            self._show_rejected(rejected)
        else:
            self._note(None)
        self.try_row.setVisible(not rows)
        self._live_since = None
        if accepted:
            self._prompt_timer.start(500)
        else:
            self._prompt_timer.stop()
        self.prompt.clearFocus()
        self.prompts_changed.emit(accepted)

    def _show_rejected(self, rejected):
        names = ", ".join(f"'{raw}'" for raw, _ in rejected)
        fixes = [(raw, s) for raw, sug in rejected for s in sug[:2]]
        text = (f"{names} is not a known object name, so it is not searched." if len(rejected) == 1 else
                f"{names} are not known object names, so they are not searched.")
        if fixes:
            text += " Did you mean:"
        self._note(text, "bad")
        if fixes:
            frow = QWidget()
            flay = FlowLayout(frow)
            for raw, fix in fixes[:4]:
                b = QPushButton(fix)
                b.setObjectName("Mini")
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setToolTip(f"Use '{fix}' instead of '{raw}'")
                b.clicked.connect(lambda _=False, r=raw, f=fix: self._replace_word(r, f))
                flay.addWidget(b)
            self.fix_box.addWidget(frow)
            self._fix_buttons.append(frow)

    def prompts_live(self):
        """Called once open-vocabulary results start arriving; starts the 'not seen yet' clock."""
        if self._live_since is None and self._accepted:
            self._live_since = time.monotonic()
            self._refresh_prompt_status()

    def _refresh_prompt_status(self):
        if not self._accepted:
            return
        waited = 0.0 if self._live_since is None else time.monotonic() - self._live_since
        missing = []
        for row in self._accepted:
            if row.count:
                row.seen = True
                row.set_status("")
            elif self._live_since is None:
                row.set_status("starting…")
            elif row.seen:
                row.set_status("found ✓", "#7CFFCB")
            elif waited < NOT_SEEN_S:
                row.set_status("looking…")
            else:
                row.set_status("not seen yet", "#FFB84D")
                missing.append(row)
        if self._note_kind == "bad":
            return  # keep the 'unknown word' message on screen
        if missing:
            self._note("YOLOE-26 recognizes some words only weakly (body parts like hair or face, "
                                     "materials, scenery). Hold the object closer, lower the confidence, or try "
                                     "another name.", "warn")
        else:
            self._note(None)

    def _note(self, text: str | None, kind: str = ""):
        self._note_kind = kind if text else ""
        if not text:
            self.prompt_note.hide()
            return
        color = {"bad": "#FF6B7A", "warn": "#FFB84D"}.get(kind, "#8a95b3")
        self.prompt_note.setStyleSheet(f"font-size: 10px; color: {color};")
        self.prompt_note.setText(text)
        self.prompt_note.show()

    def _emit_conf(self, v: int):
        self.conf_value.setText(f"{v}%")
        self.confidence_changed.emit(v / 100.0)

    def _mission_toggled(self, on: bool):
        self.mission_btn.setText("■  STOP MISSION" if on else "🎯  START MISSION")
        self.mission_toggled.emit(on)

    def set_model(self, preset: str, label: str):
        """Reflect the detector that is actually running (also used to undo a failed switch)."""
        if preset in self.model_buttons:
            self.model_buttons[preset].setChecked(True)
        self.tagline.setText(f"Real-time {label} vision · Apple GPU")

    def set_mission(self, m):
        """Live progress line under the mission button."""
        if m.active:
            self.mission_status.setText(f"Round {m.round}/{m.total} · {m.found} found · {m.score} pts")
        elif m.state == "done":
            self.mission_status.setText(f"Done: {m.found}/{m.total} found · {m.score} pts")
        else:
            self.mission_status.setText(f"{m.rounds} rounds · {m.seconds:g}s each")
        self.skip_btn.setEnabled(m.active and m.state == "hunt")

    def mission_done(self):
        """The game ended on its own: reset the button without restarting anything."""
        self.mission_btn.blockSignals(True)
        self.mission_btn.setChecked(False)
        self.mission_btn.setText("🎯  START MISSION")
        self.mission_btn.blockSignals(False)

    def set_counts(self, counts: dict):
        for r in self.rows:
            r.set_count(counts.get(r.obj.class_id, 0))
        for i, row in enumerate(self._accepted):
            row.set_count(counts.get(CUSTOM_BASE + i, 0))
            if row.count and not row.seen:
                self._refresh_prompt_status()

    def set_stats(self, fps: float, ms: float, seen: int, history):
        self.stat_fps.setText(f"{fps:.0f}")
        self.stat_ms.setText(f"{ms:.1f}")
        self.stat_seen.setText(str(seen))
        self.spark.set_values(history)
