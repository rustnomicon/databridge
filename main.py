import sys, logging
from typing import Dict, Any, Optional, List, Tuple, Iterable
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFormLayout, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QFileDialog, QTextEdit, QDialog, QProgressDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
from business_logic import CsvStream, Transformer, ClickHouseUploader, make_staging_sql
from multi_select_combobox import CheckableComboBox
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('databridge.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

APP_ORG = "Rustnomicon"
APP_NAME = "DataBridge"


class FilterDialog(QDialog):
    def __init__(self, initial: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Фильтры колонки")
        self.rules = initial.copy() if initial else {}
        lay = QFormLayout(self)
        self.ed_remove_chars = QLineEdit(",".join(self.rules.get("remove_chars", [])))
        self.ed_regex_remove = QTextEdit("\n".join(self.rules.get("regex_remove", [])))
        self.ed_regex_replace = QTextEdit(json.dumps(self.rules.get("regex_replace", []), ensure_ascii=False, indent=2))
        self.chk_trim = self._check("trim", True)
        self.chk_digits = self._check("digits_only", False)
        self.chk_normalize_phone = self._check("normalize_phone", False)
        self.chk_lower = self._check("lower", False)
        self.chk_upper = self._check("upper", False)
        self.chk_date = self._check("format_date", False)

        lay.addRow("Удалить символы (через запятую):", self.ed_remove_chars)
        lay.addRow("Regex удалить (по одному на строку):", self.ed_regex_remove)
        lay.addRow("Regex replace (JSON):", self.ed_regex_replace)
        lay.addRow("Trim", self.chk_trim)
        lay.addRow("Digits only", self.chk_digits)
        lay.addRow("Нормализация телефона (RU)", self.chk_normalize_phone)
        lay.addRow("lower()", self.chk_lower)
        lay.addRow("UPPER()", self.chk_upper)
        lay.addRow("Дата YYYY-MM-DD", self.chk_date)

        buttons = QHBoxLayout()
        ok = QPushButton("OK");
        cancel = QPushButton("Отмена")
        ok.clicked.connect(self.accept);
        cancel.clicked.connect(self.reject)
        buttons.addWidget(ok);
        buttons.addWidget(cancel)
        lay.addRow(buttons)

    def _check(self, key: str, default: bool):
        cb = QPushButton()
        cb.setCheckable(True)
        cb.setChecked(self.rules.get(key, default))
        cb.setText("✓" if cb.isChecked() else " ")
        cb.toggled.connect(lambda v, b=cb: b.setText("✓" if v else " "))
        cb.setFixedWidth(30)
        return cb

    def get_rules(self) -> Dict[str, Any]:
        rules = {
            "trim": self.chk_trim.isChecked(),
            "digits_only": self.chk_digits.isChecked(),
            "normalize_phone": self.chk_normalize_phone.isChecked(),
            "lower": self.chk_lower.isChecked(),
            "upper": self.chk_upper.isChecked(),
            "format_date": self.chk_date.isChecked()
        }
        remove_chars = [x.strip() for x in self.ed_remove_chars.text().split(",") if x.strip()]
        if remove_chars:
            rules["remove_chars"] = remove_chars
        regex_remove = [ln.strip() for ln in self.ed_regex_remove.toPlainText().splitlines() if ln.strip()]
        if regex_remove:
            rules["regex_remove"] = regex_remove
        try:
            regex_replace = json.loads(self.ed_regex_replace.toPlainText() or "[]")
            if isinstance(regex_replace, list):
                rules["regex_replace"] = regex_replace
        except Exception:
            pass
        return rules


import json
from pathlib import Path
from PySide6.QtWidgets import QMessageBox

class SettingsTab(QWidget):
    SETTINGS_FILE = Path(__file__).parent / "settings.json"

    def __init__(self):
        super().__init__()
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
            except Exception as e:
                print(f"Ошибка чтения настроек из JSON: {e}")

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
        # Создаём виджет для размещения текста и кнопки
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)  # Убираем отступы внутри виджета

        info_label = QLabel(
            "<b>Динамические значения:</b><br>"
            "Для подстановки текущей даты/времени UTC используйте в поле 'Статическое значение':<br>"
            "Формат: <code>{{CURRENT_DATETIME}}</code>"
        )
        info_label.setWordWrap(True)

        copy_button = QPushButton("Копировать")
        # Устанавливаем размер кнопки "копировать" по размеру содержимого
        copy_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Подключаем сигнал clicked к лямбде, чтобы передать нужное значение
        copy_button.clicked.connect(lambda: self.copy_to_clipboard("{{CURRENT_DATETIME}}"))

        info_layout.addWidget(info_label)
        info_layout.addWidget(copy_button)
        # Добавляем виджет с текстом и кнопкой в основной layout
        form.addRow(info_widget)

        # Стиль для основного виджета с информацией
        info_widget.setStyleSheet(
            """
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            color: #333;
            """
        )

    def copy_to_clipboard(self, text: str):
        """Копирует указанный текст в буфер обмена."""
        clipboard = QApplication.clipboard()  # type: ignore # Игнорируем ошибку линтера, QApplication гарантированно существует
        clipboard.setText(text)
        # Опционально: показать всплывающее сообщение
        # QMessageBox.information(self, "Скопировано", f"'{text}' скопировано в буфер обмена.")

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
        # Пишем в JSON
        try:
            self.SETTINGS_FILE.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            QMessageBox.information(self, "OK", "Настройки сохранены в settings.json")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить settings.json: {e}")

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

class ImportWorker(QObject):
    progress = Signal(int, float)   # total_inserted, rows_per_sec
    finished = Signal(int)          # total_inserted
    failed = Signal(str)

    def __init__(self,
                 csv_path: str,
                 mapping: Dict[str, str],
                 filters_by_csv: Dict[str, Dict[str, Any]],
                 static_values: Dict[str, str],
                 conn: Dict[str, Any],
                 batch_size: int,
                 workers: int):
        super().__init__()
        self.csv_path = csv_path
        self.mapping = mapping
        self.filters_by_csv = filters_by_csv
        self.static_values = static_values
        self.conn = conn
        self.batch_size = batch_size
        self.workers = workers

    def run(self):
        try:
            stream = CsvStream(self.csv_path)
            transformer = Transformer(self.mapping, self.filters_by_csv, self.static_values)
            uploader = ClickHouseUploader(**self.conn)

            def batches() -> Iterable[Tuple[List[str], List[List[Any]]]]:
                for csv_batch in stream.iter_rows(batch_size=self.batch_size):
                    cols, data = transformer.transform_batch(csv_batch)
                    if cols and data:
                        yield (cols, data)

            total = uploader.insert_parallel(batches(), workers=self.workers,
                                             progress_cb=lambda n, r: self.progress.emit(n, r))
            self.finished.emit(total)
        except Exception as e:
            self.failed.emit(str(e))


class ImportTab(QWidget):
    def __init__(self, settings_tab: SettingsTab):
        super().__init__()
        logger.info("Инициализация вкладки импорта")
        self.settings_tab = settings_tab
        self.csv_path: Optional[str] = None
        self.csv_headers: List[str] = []
        self.ch_columns: List[str] = []
        self.filters_by_csv: Dict[str, Dict[str, Any]] = {}
        self.static_values: Dict[str, str] = {}
        self.mapping: Dict[str, List[str]] = {}

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn_pick = QPushButton("Выбрать CSV")
        self.lbl_file = QLabel("Файл: -")
        top.addWidget(self.btn_pick);
        top.addWidget(self.lbl_file, 1)
        root.addLayout(top)

        row2 = QHBoxLayout()
        self.btn_load_hdrs = QPushButton("Подгрузить заголовки CSV")
        self.btn_load_cols = QPushButton("Подгрузить колонки ClickHouse")
        self.btn_gen_sql = QPushButton("Сгенерировать INSERT SELECT SQL")
        row2.addWidget(self.btn_load_hdrs);
        row2.addWidget(self.btn_load_cols);
        row2.addWidget(self.btn_gen_sql)
        root.addLayout(row2)

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["ClickHouse колонка", "CSV колонки", "Статическое значение", "Фильтры"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self.tbl)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Batch size:"))
        self.spin_batch = QSpinBox();
        self.spin_batch.setRange(1000, 500000);
        self.spin_batch.setValue(100000)
        self.spin_batch.setFocusPolicy(Qt.StrongFocus)
        self.spin_batch.wheelEvent = lambda event: event.ignore()

        ctrl.addWidget(self.spin_batch)
        ctrl.addWidget(QLabel("Threads:"))
        self.spin_workers = QSpinBox();
        self.spin_workers.setRange(1, 16);
        self.spin_workers.setValue(4)
        ctrl.addWidget(self.spin_workers)
        self.btn_preview = QPushButton("Предпросмотр 50")
        self.btn_import = QPushButton("Импорт")
        ctrl.addWidget(self.btn_preview);
        ctrl.addWidget(self.btn_import)
        root.addLayout(ctrl)

        self.btn_pick.clicked.connect(self.pick_csv)
        self.btn_load_hdrs.clicked.connect(self.load_csv_headers)
        self.btn_load_cols.clicked.connect(self.load_ch_columns)
        self.btn_gen_sql.clicked.connect(self.generate_sql)
        self.btn_preview.clicked.connect(self.preview)
        self.btn_import.clicked.connect(self.start_import)

    def pick_csv(self):
        logger.debug("Открытие диалога выбора файла")
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "", "CSV (*.csv);;All (*.*)")
        if path:
            self.csv_path = path
            self.lbl_file.setText(f"Файл: {path}")
            logger.info(f"Выбран файл: {path}")

    def load_csv_headers(self):
        logger.info("Загрузка заголовков CSV")
        if not self.csv_path:
            logger.warning("Попытка загрузить заголовки без выбранного файла")
            QMessageBox.warning(self, "Нет файла", "Сначала выберите CSV")
            return
        try:
            self.csv_headers = CsvStream(self.csv_path).headers()
            logger.info(f"Загружено {len(self.csv_headers)} заголовков")
            QMessageBox.information(self, "CSV", ", ".join(self.csv_headers)[:300])

            # Обновляем комбобоксы в таблице
            for row in range(self.tbl.rowCount()):
                combo: CheckableComboBox = self.tbl.cellWidget(row, 1)
                if combo:
                    combo.clear()
                    combo.addItems(self.csv_headers)
        except Exception as e:
            logger.error(f"Ошибка загрузки заголовков: {e}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", str(e))

    def load_ch_columns(self):
        logger.info("Загрузка колонок ClickHouse")
        try:
            params = self.settings_tab.conn_params()
            from clickhouse_driver import Client
            c = Client(host=params["host"], port=params["port"], user=params["user"],
                       password=params["password"], database=params["database"])
            rows = c.execute(
                "SELECT name, type FROM system.columns WHERE database=%(db)s AND table=%(tbl)s ORDER BY position",
                {"db": params["database"], "tbl": params["table"]}
            )
            self.ch_columns = [r[0] for r in rows]
            logger.info(f"Загружено {len(self.ch_columns)} колонок из ClickHouse")

            self.tbl.setRowCount(0)
            for name, typ in rows:
                r = self.tbl.rowCount()
                self.tbl.insertRow(r)

                # ClickHouse column (read-only)
                self.tbl.setItem(r, 0, QTableWidgetItem(f"{name} ({typ})"))
                self.tbl.item(r, 0).setFlags(Qt.ItemIsEnabled)

                # CSV columns dropdown с чекбоксами
                combo = CheckableComboBox()
                combo.addItems(self.csv_headers)
                self.tbl.setCellWidget(r, 1, combo)

                # Static value
                ed_const = QLineEdit()
                self.tbl.setCellWidget(r, 2, ed_const)

                # Filters button
                btn = QPushButton("Фильтры…")
                btn.clicked.connect(lambda _, row=r: self.edit_filters(row))
                self.tbl.setCellWidget(r, 3, btn)

            QMessageBox.information(self, "ClickHouse", f"Загружено {len(self.ch_columns)} колонок")
        except Exception as e:
            logger.error(f"Ошибка загрузки колонок ClickHouse: {e}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", str(e))

    def edit_filters(self, row: int):
        combo: CheckableComboBox = self.tbl.cellWidget(row, 1)
        selected = combo.checked_items()

        if not selected:
            QMessageBox.warning(self, "Выберите колонки", "Сначала выберите CSV колонки для настройки фильтров")
            return

        # Показываем фильтры для первой выбранной колонки
        csv_col = selected[0]
        logger.debug(f"Редактирование фильтров для колонки: {csv_col}")
        initial = self.filters_by_csv.get(csv_col, {})
        dlg = FilterDialog(initial, self)

        if dlg.exec():
            rules = dlg.get_rules()
            # Применяем фильтры ко всем выбранным колонкам
            for col_name in selected:
                self.filters_by_csv[col_name] = rules
                logger.info(f"Фильтры обновлены для {col_name}")

    def collect_mapping(self):
        logger.debug("Сбор маппинга колонок")
        mapping: Dict[str, List[str]] = {}
        statics: Dict[str, str] = {}

        for r in range(self.tbl.rowCount()):
            ch_label = self.tbl.item(r, 0).text()
            ch_col = ch_label.split(" ")[0]

            # Получаем выбранные CSV колонки из комбобокса
            combo: CheckableComboBox = self.tbl.cellWidget(r, 1)
            csv_cols = combo.checked_items() if combo else []

            # Получаем статическое значение напрямую из QLineEdit
            const_widget = self.tbl.cellWidget(r, 2)
            const_val = ""
            if isinstance(const_widget, QLineEdit):
                const_val = const_widget.text().strip()

            if csv_cols:
                mapping[ch_col] = csv_cols
                logger.debug(f"Маппинг {ch_col} ← {csv_cols}")

            if const_val:
                statics[ch_col] = const_val
                logger.debug(f"Статическое значение {ch_col} = {const_val}")

        self.mapping = mapping
        self.static_values = statics
        logger.info(f"Маппинг собран: {len(mapping)} маппингов, {len(statics)} статических значений")

    def preview(self):
        logger.info("Предпросмотр данных")
        if not self.csv_path:
            QMessageBox.warning(self, "Нет файла", "Выберите CSV")
            return
        self.collect_mapping()
        try:
            stream = CsvStream(self.csv_path)
            transformer = Transformer(self.mapping, self.filters_by_csv, self.static_values)
            rows = []
            it = stream.iter_rows(batch_size=50)
            first_batch = next(it, [])
            cols, data = transformer.transform_batch(first_batch)
            for r in data[:50]:
                rows.append(dict(zip(cols, r)))
            preview_text = json.dumps(rows, ensure_ascii=False, indent=2)[:4000]
            logger.debug(f"Предпросмотр: {len(rows)} строк")
            QMessageBox.information(self, "Предпросмотр", preview_text)
        except Exception as e:
            logger.error(f"Ошибка предпросмотра: {e}", exc_info=True)
            QMessageBox.critical(self, "Ошибка", str(e))

    def generate_sql(self):
        logger.info("Генерация SQL")
        self.collect_mapping()
        params = self.settings_tab.conn_params()
        if not params["table"]:
            logger.warning("Попытка генерации SQL без указания таблицы")
            QMessageBox.warning(self, "Нет таблицы", "Укажите TABLE в настройках")
            return
        staging_table = params["table"] + "_staging"
        sql = make_staging_sql(params["database"], staging_table, params["database"], params["table"],
                               self.mapping, self.filters_by_csv, self.static_values)
        dlg = QDialog(self)
        dlg.setWindowTitle("INSERT SELECT SQL")
        lay = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setPlainText(sql)
        lay.addWidget(txt)
        btn = QPushButton("Закрыть")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.resize(900, 600)
        dlg.exec()

    def start_import(self):
        logger.info("Начало импорта")
        if not self.csv_path:
            logger.warning("Попытка импорта без выбранного файла")
            QMessageBox.warning(self, "Нет файла", "Выберите CSV")
            return
        self.collect_mapping()
        params = self.settings_tab.conn_params()
        if not params["table"]:
            logger.warning("Попытка импорта без указания таблицы")
            QMessageBox.warning(self, "Нет таблицы", "Укажите TABLE в настройках")
            return

        self.progress = QProgressDialog("Импорт...", "Отмена", 0, 0, self)
        self.progress.setWindowModality(Qt.ApplicationModal)
        self.progress.setMinimumDuration(0)
        self.progress.show()

        self.thread = QThread()
        self.worker = ImportWorker(
            csv_path=self.csv_path,
            mapping=self.mapping,
            filters_by_csv=self.filters_by_csv,
            static_values=self.static_values,
            conn=params,
            batch_size=self.spin_batch.value(),
            workers=self.spin_workers.value()
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        logger.info("Фоновый поток импорта запущен")

    @Slot(int, float)
    def on_progress(self, total: int, rate: float):
        self.progress.setLabelText(f"Вставлено: {total} строк, скорость: {rate:,.0f} строк/сек")
        QApplication.processEvents()

    @Slot(int)
    def on_finished(self, total: int):
        self.progress.close()
        logger.info(f"Импорт завершён успешно: {total} строк")
        QMessageBox.information(self, "Готово", f"Загружено строк: {total}")

    @Slot(str)
    def on_failed(self, msg: str):
        self.progress.close()
        logger.error(f"Импорт завершился с ошибкой: {msg}")
        QMessageBox.critical(self, "Ошибка", msg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataBridge")
        tabs = QTabWidget()
        self.settings = SettingsTab()
        self.import_tab = ImportTab(self.settings)
        tabs.addTab(self.import_tab, "Импорт")
        tabs.addTab(self.settings, "Настройки")
        self.setCentralWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 760)
    win.show()
    sys.exit(app.exec())
