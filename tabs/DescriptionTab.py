from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt

class DescriptionTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # HTML текст с описанием
        full_description_text = (
            "<h3 align='center'>Описание</h3>"
            "<b>Динамические значения:</b><br>"
            "Для подстановки текущей даты/времени UTC используйте в поле 'Статическое значение':<br>"
            "Формат: <code>{{CURRENT_DATETIME}}</code><br><br>"

            "<b>Шаблоны для форматирования:</b><br>"
            "Для использования значений из CSV колонок в статическом поле:<br>"
            "1. Выберите нужные CSV колонки в столбце 'CSV колонки'<br>"
            "2. В поле 'Статическое значение' используйте: <code>{0}</code>, <code>{1}</code>, <code>{2}</code>...<br>"
            "Пример: <code>Имя: {0}, Фамилия: {1}</code>"
        )

        label = QLabel(full_description_text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setOpenExternalLinks(True)

        # Добавляем виджет с выравниванием по верху
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)
