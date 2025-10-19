from PySide6.QtWidgets import QComboBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem

class CheckableComboBox(QComboBox):
    """QComboBox с чекбоксами для множественного выбора"""
    selectionChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)

        # Отключаем изменение через колесико мыши
        self.setFocusPolicy(Qt.StrongFocus)

        # Модель для элементов
        self.setModel(QStandardItemModel(self))

        # Обработчик клика
        self.view().pressed.connect(self.on_item_pressed)

    def wheelEvent(self, event):
        """Отключаем прокрутку колесиком мыши"""
        event.ignore()

    def addItems(self, items):
        """Добавляет элементы с чекбоксами"""
        for text in items:
            item = QStandardItem(text)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            self.model().appendRow(item)
        self.update_display_text()

    def on_item_pressed(self, index):
        """Переключает состояние чекбокса при клике"""
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        self.update_display_text()
        self.selectionChanged.emit(self.checked_items())

    def checked_items(self):
        """Возвращает список выбранных элементов"""
        checked = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item and item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def update_display_text(self):
        """Обновляет текст в поле комбобокса"""
        checked = self.checked_items()
        if checked:
            text = ", ".join(checked)
        else:
            text = "Выберите колонки..."
        self.lineEdit().setText(text)
        self.lineEdit().home(False)  # Прокрутка к началу

    def clear(self):
        """Очищает все элементы"""
        self.model().clear()
        self.update_display_text()

    def set_checked_items(self, items):
        """Устанавливает выбранные элементы"""
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item:
                if item.text() in items:
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
        self.update_display_text()