import json, logging
from typing import Optional, List, Dict, Any, Iterable, Tuple

from PySide6.QtCore import Qt, QThread, Slot, QObject, Signal
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget, QHeaderView, \
    QSpinBox, QFileDialog, QMessageBox, QTableWidgetItem, QLineEdit, QDialog, QTextEdit, QProgressDialog, QApplication, \
    QComboBox

from business_logic import CsvStream, Transformer, make_staging_sql, ClickHouseUploader
from dialogs.FilterDialog import FilterDialog
from multi_select_combobox import CheckableComboBox
from tabs.SettingsTab import SettingsTab

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('databridge.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ZoomableTableWidget(QTableWidget):
    """QTableWidget с поддержкой зума через Ctrl+колесико и Ctrl+/-"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zoom_factor = 1.0
        self._original_font = self.font()
        self._min_zoom = 0.5
        self._max_zoom = 3.0

    def _apply_zoom(self):
        """Применяет зум с блокировкой обновлений для производительности"""
        self.setUpdatesEnabled(False)

        new_font = self.font()
        new_font.setPointSizeF(self._original_font.pointSizeF() * self._zoom_factor)
        self.setFont(new_font)

        # Применяем шрифт к заголовкам для единообразия
        self.horizontalHeader().setFont(new_font)
        self.verticalHeader().setFont(new_font)

        self.setUpdatesEnabled(True)

    def wheelEvent(self, event: QWheelEvent):
        # Проверяем, зажата ли клавиша Ctrl
        if event.modifiers() == Qt.ControlModifier:
            # Определяем направление вращения колеса
            delta = event.angleDelta().y()

            if delta > 0:
                # Прокрутка вверх (от пользователя) = увеличение
                self._zoom_factor *= 1.1
            else:
                # Прокрутка вниз (к пользователю) = уменьшение
                self._zoom_factor /= 1.1

            # Ограничиваем диапазон зума
            self._zoom_factor = max(self._min_zoom, min(self._max_zoom, self._zoom_factor))

            # Применяем зум
            self._apply_zoom()

            event.accept()
        else:
            # Если Ctrl не зажат, передаем событие дальше (для прокрутки)
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        # Проверяем, зажата ли клавиша Ctrl
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
                # Увеличение (Ctrl + или Ctrl =)
                self._zoom_factor *= 1.1
                self._zoom_factor = min(self._max_zoom, self._zoom_factor)
                self._apply_zoom()
                event.accept()
            elif event.key() == Qt.Key_Minus:
                # Уменьшение (Ctrl -)
                self._zoom_factor /= 1.1
                self._zoom_factor = max(self._min_zoom, self._zoom_factor)
                self._apply_zoom()
                event.accept()
            elif event.key() == Qt.Key_0:
                # Сброс зума (Ctrl + 0)
                self._zoom_factor = 1.0
                self._apply_zoom()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            # Если Ctrl не зажат, передаем событие дальше
            super().keyPressEvent(event)


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
        self.lbl_file = QLabel("Файл: -")  # Обычный QLabel

        # Добавляем выбор разделителя
        top.addWidget(QLabel("Разделитель:"))
        self.combo_delimiter = QComboBox()
        self.combo_delimiter.addItems([
            "Запятая (,)",
            "Точка с запятой (;)",
            "Табуляция (\\t)",
            "Вертикальная черта (|)",
            "Двоеточие (:)",
            "Пользовательский"
        ])
        self.combo_delimiter.setCurrentIndex(0)
        self.combo_delimiter.currentIndexChanged.connect(self.on_delimiter_changed)
        top.addWidget(self.combo_delimiter)

        # Поле для пользовательского разделителя
        self.line_custom_delimiter = QLineEdit()
        self.line_custom_delimiter.setMaximumWidth(50)
        self.line_custom_delimiter.setPlaceholderText("...")
        self.line_custom_delimiter.setVisible(False)
        top.addWidget(self.line_custom_delimiter)

        top.addWidget(self.btn_pick)
        top.addWidget(self.lbl_file, 1)
        root.addLayout(top)

        row2 = QHBoxLayout()
        self.btn_load_hdrs = QPushButton("Подгрузить заголовки CSV")
        self.btn_load_cols = QPushButton("Подгрузить колонки ClickHouse")
        self.btn_gen_sql = QPushButton("Сгенерировать INSERT SELECT SQL")
        row2.addWidget(self.btn_load_hdrs)
        row2.addWidget(self.btn_load_cols)
        row2.addWidget(self.btn_gen_sql)
        root.addLayout(row2)

        # Используем ZoomableTableWidget вместо QTableWidget
        self.tbl = ZoomableTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["ClickHouse колонка", "CSV колонки", "Статическое значение", "Фильтры"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self.tbl)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Batch size:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1000, 500000)
        self.spin_batch.setValue(100000)
        self.spin_batch.setFocusPolicy(Qt.StrongFocus)
        self.spin_batch.wheelEvent = lambda event: event.ignore()

        ctrl.addWidget(self.spin_batch)
        ctrl.addWidget(QLabel("Threads:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 16)
        self.spin_workers.setValue(4)
        ctrl.addWidget(self.spin_workers)
        self.btn_preview = QPushButton("Предпросмотр 50")
        self.btn_import = QPushButton("Импорт")
        ctrl.addWidget(self.btn_preview)
        ctrl.addWidget(self.btn_import)
        root.addLayout(ctrl)

        self.btn_pick.clicked.connect(self.pick_csv)
        self.btn_load_hdrs.clicked.connect(self.load_csv_headers)
        self.btn_load_cols.clicked.connect(self.load_ch_columns)
        self.btn_gen_sql.clicked.connect(self.generate_sql)
        self.btn_preview.clicked.connect(self.preview)
        self.btn_import.clicked.connect(self.start_import)

    def on_delimiter_changed(self):
        """Показывает/скрывает поле для пользовательского разделителя"""
        is_custom = self.combo_delimiter.currentText() == "Пользовательский"
        self.line_custom_delimiter.setVisible(is_custom)
        logger.debug(f"Выбран разделитель: {self.combo_delimiter.currentText()}")

    def get_delimiter(self) -> str:
        """Возвращает текущий разделитель на основе выбора пользователя"""
        delimiter_map = {
            "Запятая (,)": ",",
            "Точка с запятой (;)": ";",
            "Табуляция (\\t)": "\t",
            "Вертикальная черта (|)": "|",
            "Двоеточие (:)": ":"
        }

        selected = self.combo_delimiter.currentText()
        if selected == "Пользовательский":
            custom = self.line_custom_delimiter.text()
            return custom if custom else ","

        return delimiter_map.get(selected, ",")

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
            delimiter = self.get_delimiter()
            self.csv_headers = CsvStream(self.csv_path, delimiter=delimiter).headers()
            logger.info(f"Загружено {len(self.csv_headers)} заголовков с разделителем '{delimiter}'")
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
            delimiter = self.get_delimiter()
            stream = CsvStream(self.csv_path, delimiter=delimiter)
            transformer = Transformer(self.mapping, self.filters_by_csv, self.static_values)
            rows = []
            it = stream.iter_rows(batch_size=50)
            first_batch = next(it, [])
            cols, data = transformer.transform_batch(first_batch)
            for r in data[:50]:
                # Преобразуем все значения в строку перед созданием словаря
                str_row = {col: str(val) for col, val in zip(cols, r)}
                rows.append(str_row)
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

        delimiter = self.get_delimiter()
        self.thread = QThread()
        self.worker = ImportWorker(
            csv_path=self.csv_path,
            delimiter=delimiter,
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


class ImportWorker(QObject):
    progress = Signal(int, float)  # total_inserted, rows_per_sec
    finished = Signal(int)  # total_inserted
    failed = Signal(str)

    def __init__(self,
                 csv_path: str,
                 delimiter: str,
                 mapping: Dict[str, str],
                 filters_by_csv: Dict[str, Dict[str, Any]],
                 static_values: Dict[str, str],
                 conn: Dict[str, Any],
                 batch_size: int,
                 workers: int):
        super().__init__()
        self.csv_path = csv_path
        self.delimiter = delimiter
        self.mapping = mapping
        self.filters_by_csv = filters_by_csv
        self.static_values = static_values
        self.conn = conn
        self.batch_size = batch_size
        self.workers = workers

    def run(self):
        try:
            stream = CsvStream(self.csv_path, delimiter=self.delimiter)
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
