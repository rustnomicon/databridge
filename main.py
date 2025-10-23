import sys, logging
from typing import Dict, Any, List, Tuple, Iterable
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget
)
from PySide6.QtCore import Signal, QObject

from tabs.DescriptionTab import DescriptionTab
from tabs.ImportTab import ImportTab
from tabs.SettingsTab import SettingsTab
from business_logic import CsvStream, Transformer, ClickHouseUploader

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

APP_ORG = "Rustnomicon"
APP_NAME = "DataBridge"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataBridge")
        tabs = QTabWidget()
        self.settings = SettingsTab()
        self.import_tab = ImportTab(self.settings)
        self.descriptionTab = DescriptionTab()
        tabs.addTab(self.import_tab, "Импорт")
        tabs.addTab(self.settings, "Настройки")
        tabs.addTab(self.descriptionTab, "Справочник")
        self.setCentralWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 760)
    win.show()
    sys.exit(app.exec())
