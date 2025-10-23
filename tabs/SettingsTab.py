import json
import sys
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import QStandardPaths, QCoreApplication
from PySide6.QtWidgets import QWidget, QFormLayout, QLineEdit, QHBoxLayout, QPushButton, QLabel, QSizePolicy, \
    QApplication, QMessageBox


class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()

        # Устанавливаем имя приложения для QStandardPaths
        if not QCoreApplication.organizationName():
            QCoreApplication.setOrganizationName("DataBridge")
        if not QCoreApplication.applicationName():
            QCoreApplication.setApplicationName("DataBridge")

        # Определяем путь к файлу настроек
        self.SETTINGS_FILE = self._get_settings_path()

        # Попробуем загрузить из JSON
        self.data = {
            "host": "localhost",
            "port": "9000",
            "user": "default",
            "password": "",
            "database": "default",
            "table": ""
        }

        if self.SETTINGS_FILE.exists():
            try:
                loaded = json.loads(self.SETTINGS_FILE.read_text(encoding="utf-8"))
                self.data.update(loaded)
                print(f"Настройки загружены из: {self.SETTINGS_FILE}")
            except Exception as e:
                print(f"Ошибка чтения настроек из JSON: {e}")
        else:
            print(f"Файл настроек будет создан в: {self.SETTINGS_FILE}")

        form = QFormLayout(self)
        self.ed_host = QLineEdit(self.data["host"])
        self.ed_port = QLineEdit(self.data["port"])
        self.ed_user = QLineEdit(self.data["user"])
        self.ed_password = QLineEdit(self.data["password"])
        self.ed_password.setEchoMode(QLineEdit.Password)
        self.ed_db = QLineEdit(self.data["database"])
        self.ed_table = QLineEdit(self.data["table"])

        form.addRow("HOST:", self.ed_host)
        form.addRow("PORT:", self.ed_port)
        form.addRow("USER:", self.ed_user)
        form.addRow("PASSWORD:", self.ed_password)
        form.addRow("DATABASE:", self.ed_db)
        form.addRow("TABLE:", self.ed_table)

        btns = QHBoxLayout()
        self.btn_save = QPushButton("Сохранить")
        btns.addWidget(self.btn_save)
        form.addRow(btns)
        self.btn_save.clicked.connect(self.save)

        # Информация о динамических значениях
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)

        info_label = QLabel(
            "<b>Динамические значения:</b><br>"
            "Для подстановки текущей даты/времени UTC используйте в поле 'Статическое значение':<br>"
            "Формат: <code>{{CURRENT_DATETIME}}</code>"
        )
        info_label.setWordWrap(True)

        copy_button = QPushButton("Копировать")
        copy_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        copy_button.clicked.connect(lambda: self.copy_to_clipboard("{{CURRENT_DATETIME}}"))

        info_layout.addWidget(info_label)
        info_layout.addWidget(copy_button)
        form.addRow(info_widget)

        info_widget.setStyleSheet(
            """
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            color: #333;
            """
        )

    def _get_settings_path(self) -> Path:
        """
        Определяет корректный путь для сохранения настроек.
        Использует QStandardPaths для кросс-платформенной совместимости.
        """
        # Для PyInstaller и обычного запуска
        if getattr(sys, 'frozen', False):
            # Приложение запущено из exe (PyInstaller)
            # Используем AppConfigLocation для настроек
            config_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
        else:
            # Обычный запуск из исходников
            # Сохраняем рядом с файлом для удобства разработки
            config_dir = str(Path(__file__).parent)

        # Создаём директорию, если её нет
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        return config_path / "settings.json"

    def copy_to_clipboard(self, text: str):
        """Копирует указанный текст в буфер обмена."""
        clipboard = QApplication.clipboard()  # type: ignore
        clipboard.setText(text)

    def save(self):
        # Собираем из полей
        self.data = {
            "host": self.ed_host.text().strip(),
            "port": self.ed_port.text().strip(),
            "user": self.ed_user.text().strip(),
            "password": self.ed_password.text(),
            "database": self.ed_db.text().strip(),
            "table": self.ed_table.text().strip()
        }

        # Создаём директорию, если её нет
        try:
            self.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.SETTINGS_FILE.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            QMessageBox.information(
                self,
                "OK",
                f"Настройки сохранены в:\n{self.SETTINGS_FILE}"
            )
            print(f"Настройки успешно сохранены в: {self.SETTINGS_FILE}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось сохранить settings.json:\n{e}\n\nПуть: {self.SETTINGS_FILE}"
            )
            print(f"Ошибка сохранения настроек: {e}")

    def conn_params(self) -> Dict[str, Any]:
        # Берём из self.data, приводим port к int
        return {
            "host": self.data["host"],
            "port": int(self.data.get("port") or 9000),
            "user": self.data["user"],
            "password": self.data["password"],
            "database": self.data["database"],
            "table": self.data["table"]
        }
