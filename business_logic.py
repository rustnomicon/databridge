import csv, re, time, threading, logging
from typing import Dict, Any, Iterable, List, Callable, Optional, Tuple, LiteralString
from clickhouse_driver import Client

# Настройка логирования
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

import datetime

# Единственный плейсхолдер для текущей даты
DYNAMIC_PLACEHOLDERS = {
    "{{CURRENT_DATETIME}}": lambda: datetime.datetime.now(datetime.UTC),
}

# SQL эквивалент для ClickHouse
SQL_PLACEHOLDERS = {
    "{{CURRENT_DATETIME}}": "now64(3)",
}


def resolve_dynamic_value(value: str) -> str:
    """Заменяет {{CURRENT_DATETIME}} на реальное значение"""
    if value in DYNAMIC_PLACEHOLDERS:
        return DYNAMIC_PLACEHOLDERS[value]()
    return value


# Правила трансформаций на Python-стороне
def safe_parse_date(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            result = datetime.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            logger.debug(f"Дата '{s}' распознана форматом {fmt} → {result}")
            return result
        except ValueError:
            continue
    logger.warning(f"Не удалось распознать дату: '{s}'")
    return s


def normalize_phone(phone: str) -> int:
    """Нормализует телефонный номер к формату 7XXXXXXXXXX и возвращает int"""
    phone = phone.strip()
    # Удалить все нецифровые символы
    digits = re.sub(r'\D', '', phone)

    # Убрать первую "8" или "7" (если номер российский)
    if digits.startswith('8'):
        digits = '7' + digits[1:]
    elif digits.startswith('7') and len(digits) == 11:
        pass  # уже в нужном виде
    elif digits.startswith('007') and len(digits) == 12:
        digits = '7' + digits[3:]
    elif len(digits) == 10:  # Номер без кода страны
        digits = '7' + digits

    # Если получилось 11 цифр с первой "7" — возвращаем как int, иначе 0
    if len(digits) == 11 and digits.startswith('7'):
        return int(digits)
    else:
        logger.warning(f"Не удалось нормализовать телефон: '{phone}' → '{digits}'")
        return 0  # Возвращаем 0 как int для невалидных номеров


def apply_filters_py(value: Any, rules: Dict[str, Any]) -> int | str | bytes | LiteralString:
    original = "" if value is None else str(value)
    s = original

    if rules.get("trim", True):
        s = s.strip()
    if rules.get("remove_chars"):
        for ch in rules["remove_chars"]:
            s = s.replace(ch, "")
    if rules.get("regex_remove"):
        for pat in rules["regex_remove"]:
            s = re.sub(pat, "", s)
    if rules.get("regex_replace"):
        for pair in rules["regex_replace"]:
            try:
                s = re.sub(pair.get("pattern", ""), pair.get("repl", ""), s)
            except re.error as e:
                logger.error(f"Ошибка regex: {e}, pattern={pair.get('pattern')}")

    # ВАЖНО: digits_only теперь возвращает int, если указан to_integer
    if rules.get("digits_only"):
        s = re.sub(r"\D+", "", s)
        # Если также требуется to_integer, сразу конвертируем
        if rules.get("to_integer"):
            if s:
                s = int(s)
            else:
                logger.warning(f"Нет цифр в значении '{original}', установлено в 0")
                s = 0

    # normalize_phone уже возвращает int
    if rules.get("normalize_phone"):
        s = normalize_phone(str(s))  # Гарантируем, что передается строка
        # normalize_phone всегда возвращает int, дополнительная проверка не нужна

    if rules.get("lower"):
        s = s.lower() if isinstance(s, str) else s
    if rules.get("upper"):
        s = s.upper() if isinstance(s, str) else s
    if rules.get("format_date"):
        s = safe_parse_date(str(s))

    if rules.get("to_string"):
        if isinstance(s, (int, float)):
            s = str(s)
        elif isinstance(s, datetime.datetime):
            s = s.strftime(rules.get("format", "%Y-%m-%d %H:%M:%S"))
        else:
            s = str(s)

    # to_integer обрабатываем, только если еще не int (и не обработан digits_only)
    if rules.get("to_integer") and not isinstance(s, int):
        try:
            # Преобразуем в строку если это не строка
            str_value = str(s) if s is not None else ""

            # Убираем ВСЕ символы кроме цифр
            digits = re.sub(r"\D", "", str_value)

            # Проверяем что остались цифры после очистки
            if digits:
                s = int(digits)
            else:
                logger.warning(f"Нет цифр в значении '{original}', установлено в 0")
                s = 0
        except (ValueError, AttributeError) as e:
            logger.warning(f"Ошибка конвертации в int для '{original}': {e}, установлено в 0")
            s = 0

    # ФИНАЛЬНАЯ ПРОВЕРКА: если normalize_phone или to_integer, гарантируем int
    if (rules.get("normalize_phone") or rules.get("to_integer")) and not isinstance(s, int):
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: ожидался int, получен {type(s).__name__}: '{s}' (исходное: '{original}')")
        s = 0

    if original != s:
        logger.debug(f"Фильтры: '{str(original)[:50]}' → '{str(s)[:50]}' (тип: {type(s).__name__})")

    return s


class CsvStream:
    """Класс для работы с CSV файлами с поддержкой пользовательского разделителя"""

    def __init__(self, path: str, delimiter: str = ',', encoding: str = "utf-8-sig"):
        """
        Инициализация CsvStream

        Args:
            path: путь к CSV файлу
            delimiter: разделитель полей (по умолчанию запятая)
            encoding: кодировка файла (по умолчанию utf-8-sig)
        """
        self.path = path
        self.delimiter = delimiter
        self.encoding = encoding
        logger.info(f"Инициализация CsvStream: {path}, delimiter='{delimiter}', encoding={encoding}")

    def headers(self) -> List[str]:
        """Возвращает заголовки CSV файла"""
        logger.debug(f"Чтение заголовков из {self.path}")
        try:
            with open(self.path, "r", encoding=self.encoding, newline="") as f:
                rdr = csv.reader(f, delimiter=self.delimiter)
                first = next(rdr, [])
                headers = [h.strip() for h in first] if first else []
                logger.info(f"Найдено {len(headers)} заголовков: {headers[:10]}")
                return headers
        except Exception as e:
            logger.error(f"Ошибка чтения заголовков: {e}", exc_info=True)
            raise

    def iter_rows(self, batch_size: int = 100000) -> Iterable[List[Dict[str, Any]]]:
        """
        Итератор по строкам CSV файла с группировкой в батчи

        Args:
            batch_size: размер батча

        Yields:
            Список словарей, где ключи - заголовки, значения - данные строки
        """
        logger.info(f"Начало чтения CSV {self.path}, batch_size={batch_size}, delimiter='{self.delimiter}'")
        try:
            with open(self.path, "r", encoding=self.encoding, newline="") as f:
                rdr = csv.DictReader(f, delimiter=self.delimiter)
                batch: List[Dict[str, Any]] = []
                total_rows = 0
                batch_num = 0

                for row in rdr:
                    batch.append(row)
                    total_rows += 1

                    if len(batch) >= batch_size:
                        batch_num += 1
                        logger.debug(f"Батч #{batch_num}: {len(batch)} строк, всего прочитано {total_rows}")
                        yield batch
                        batch = []

                if batch:
                    batch_num += 1
                    logger.debug(f"Финальный батч #{batch_num}: {len(batch)} строк")
                    yield batch

                logger.info(f"Чтение завершено: {total_rows} строк в {batch_num} батчах")
        except Exception as e:
            logger.error(f"Ошибка чтения CSV: {e}", exc_info=True)
            raise


class Transformer:
    def __init__(self, mapping: Dict[str, List[str]], filters_by_csv: Dict[str, Dict[str, Any]],
                 static_values: Dict[str, str]):
        self.mapping = mapping
        self.filters_by_csv = filters_by_csv
        self.static_values = static_values
        logger.info(f"Инициализация Transformer: {len(mapping)} маппингов, {len(static_values)} статических значений")
        logger.debug(f"Статические значения: {self.static_values}")

    def transform_batch(self, batch_csv: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Any]]]:
        if not self.mapping and not self.static_values:
            logger.warning("Пустой маппинг и статические значения!")
            return [], []

        cols = sorted(set(list(self.mapping.keys()) + list(self.static_values.keys())))
        out: List[List[Any]] = []

        logger.debug(f"Трансформация батча: {len(batch_csv)} строк → {len(cols)} колонок")

        for idx, row in enumerate(batch_csv):
            obj: Dict[str, Any] = {}

            # Обрабатываем маппинги
            for ch_col, csv_cols in self.mapping.items():
                if not isinstance(csv_cols, list):
                    csv_cols = [csv_cols]

                if len(csv_cols) == 1:
                    raw = row.get(csv_cols[0], "")
                    rules = self.filters_by_csv.get(csv_cols[0], {})
                    obj[ch_col] = apply_filters_py(raw, rules)
                else:
                    parts = []
                    for csv_col in csv_cols:
                        raw = row.get(csv_col, "")
                        rules = self.filters_by_csv.get(csv_col, {})
                        filtered = apply_filters_py(raw, rules)
                        if filtered is not None and filtered != "":
                            parts.append(str(filtered))
                    obj[ch_col] = " ".join(parts)

            # Обрабатываем статические значения с шаблонами
            for ch_col, val in self.static_values.items():
                # Проверяем, есть ли плейсхолдеры {0}, {1} и т.д.
                if '{' in val and '}' in val:
                    # Получаем список CSV колонок для этого ClickHouse поля
                    csv_cols = self.mapping.get(ch_col, [])
                    if not isinstance(csv_cols, list):
                        csv_cols = [csv_cols]

                    # Собираем значения из CSV колонок с применением фильтров
                    template_values = []
                    for csv_col in csv_cols:
                        raw = row.get(csv_col, "")
                        rules = self.filters_by_csv.get(csv_col, {})
                        filtered = apply_filters_py(raw, rules)
                        template_values.append(str(filtered) if filtered is not None else "")

                    # Форматируем шаблон
                    try:
                        obj[ch_col] = val.format(*template_values)
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Ошибка форматирования шаблона '{val}' для {ch_col}: {e}")
                        obj[ch_col] = val
                else:
                    # Резолвим динамические плейсхолдеры типа {{CURRENT_DATETIME}}
                    obj[ch_col] = resolve_dynamic_value(val)

            out.append([obj.get(c, None) for c in cols])

            if idx == 0:
                logger.debug(f"Первая строка после трансформации: {dict(zip(cols, out[0]))}")

        logger.debug(f"Трансформировано {len(out)} строк")
        return cols, out


class ClickHouseUploader:
    def __init__(self, host: str, port: int, user: str, password: str, database: str, table: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        logger.info(f"Инициализация ClickHouseUploader: {host}:{port}/{database}.{table}, user={user}")

    def _insert_chunk(self, cols: List[str], data_rows: List[List[Any]]) -> int:
        thread_id = threading.get_ident()
        logger.debug(f"[Thread-{thread_id}] Вставка чанка: {len(data_rows)} строк, колонки={cols}")

        try:
            client = Client(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )

            start = time.time()
            client.execute(
                f"INSERT INTO {self.database}.{self.table} ({', '.join(cols)}) VALUES",
                data_rows
            )
            duration = time.time() - start

            logger.info(
                f"[Thread-{thread_id}] Вставлено {len(data_rows)} строк за {duration:.2f}с ({len(data_rows) / duration:.0f} строк/сек)")
            return len(data_rows)

        except Exception as e:
            logger.error(f"[Thread-{thread_id}] Ошибка вставки: {e}", exc_info=True)
            raise

    def insert_parallel(
            self,
            batches: Iterable[Tuple[List[str], List[List[Any]]]],
            workers: int,
            progress_cb: Optional[Callable[[int, float], None]] = None
    ) -> int:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(f"Начало параллельной вставки: {workers} потоков")
        total = 0
        start = time.time()
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            batch_num = 0

            for cols, rows in batches:
                if not rows:
                    continue
                batch_num += 1
                logger.debug(f"Отправка батча #{batch_num} в очередь: {len(rows)} строк")
                futures.append(ex.submit(self._insert_chunk, cols, rows))

            logger.info(f"Всего батчей в очереди: {len(futures)}")

            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    inserted = fut.result()
                    with lock:
                        total += inserted
                        elapsed = time.time() - start
                        rate = total / max(0.001, elapsed)

                    logger.info(
                        f"Прогресс: {idx}/{len(futures)} батчей завершено, всего {total} строк, {rate:.0f} строк/сек")

                    if progress_cb:
                        progress_cb(total, rate)

                except Exception as e:
                    logger.error(f"Ошибка в батче {idx}: {e}", exc_info=True)

        duration = time.time() - start
        logger.info(
            f"Импорт завершён: {total} строк за {duration:.2f}с (средняя скорость: {total / duration:.0f} строк/сек)")
        return total


def make_staging_sql(
        staging_db: str,
        staging_table: str,
        target_db: str,
        target_table: str,
        mapping: Dict[str, List[str]],
        filters_by_csv: Dict[str, Dict[str, Any]],
        static_values: Dict[str, str]
) -> str:
    logger.info(f"Генерация SQL: {staging_db}.{staging_table} → {target_db}.{target_table}")

    target_cols = sorted(set(list(mapping.keys()) + list(static_values.keys())))

    def sql_expr_for(csv_col: str, rules: Dict[str, Any]) -> str:
        expr = f'`{csv_col}`'

        if rules.get("trim", True):
            expr = f"trim({expr})"
        if rules.get("remove_chars"):
            for ch in rules["remove_chars"]:
                escaped_ch = ch.replace("'", "''")
                expr = f"replaceAll({expr}, '{escaped_ch}', '')"
        if rules.get("regex_remove"):
            for pat in rules["regex_remove"]:
                escaped_pat = pat.replace("'", "''")
                expr = f"replaceRegexpAll({expr}, '{escaped_pat}', '')"
        if rules.get("regex_replace"):
            for pair in rules["regex_replace"]:
                pat = pair.get("pattern", "")
                repl = pair.get("repl", "")
                escaped_pat = pat.replace("'", "''")
                escaped_repl = repl.replace("'", "''")
                expr = f"replaceRegexpAll({expr}, '{escaped_pat}', '{escaped_repl}')"
        if rules.get("digits_only"):
            expr = f"replaceRegexpAll({expr}, '[^0-9]', '')"
        if rules.get("to_int"):
            digits_expr = f"replaceRegexpAll({expr}, '[^0-9]', '')"
            expr = f"CASE WHEN length({digits_expr}) > 0 THEN toInt32OrZero({digits_expr}) ELSE 0 END"
        if rules.get("to_string"):
            expr = f"toString({expr})"
        if rules.get("normalize_phone"):
            digits = f"replaceRegexpAll({expr}, '[^0-9]', '')"
            expr = f"""
            CASE
                WHEN startsWith({digits}, '8') AND length({digits}) = 11 
                    THEN concat('7', substring({digits}, 2))
                WHEN startsWith({digits}, '007') AND length({digits}) = 12 
                    THEN concat('7', substring({digits}, 4))
                WHEN length({digits}) = 10 
                    THEN concat('7', {digits})
                WHEN startsWith({digits}, '7') AND length({digits}) = 11 
                    THEN {digits}
                ELSE '0'
            END
            """.strip()
        if rules.get("lower"):
            expr = f"lower({expr})"
        if rules.get("upper"):
            expr = f"upper({expr})"
        if rules.get("format_date"):
            expr = f"toDate(parseDateTimeBestEffortOrNull({expr}))"

        return expr

    select_items = []

    for col in target_cols:
        if col in static_values:
            val = static_values[col]

            # Проверяем наличие шаблонов {0}, {1} и т.д.
            if '{' in val and '}' in val and not val.startswith('{{'):
                # Получаем CSV колонки для этого поля
                csv_cols = mapping.get(col, [])
                if not isinstance(csv_cols, list):
                    csv_cols = [csv_cols]

                # Создаём SQL выражения для каждой колонки
                col_exprs = []
                for csv_col in csv_cols:
                    rules = filters_by_csv.get(csv_col, {})
                    col_exprs.append(sql_expr_for(csv_col, rules))

                # Заменяем {0}, {1} на toString() выражения для concat
                template_sql = val
                for i, expr in enumerate(col_exprs):
                    # Экранируем одинарные кавычки внутри шаблона
                    template_sql = template_sql.replace(f'{{{i}}}', f"' || toString({expr}) || '")

                # Очищаем лишние concatenations
                template_sql = f"concat('{template_sql}')"
                template_sql = template_sql.replace("'' || ", "").replace(" || ''", "")

                select_items.append(f"    {template_sql} AS `{col}`")

            elif val in SQL_PLACEHOLDERS:
                select_items.append(f"    {SQL_PLACEHOLDERS[val]} AS `{col}`")
            else:
                escaped_val = val.replace("'", "''")
                select_items.append(f"    '{escaped_val}' AS `{col}`")
        else:
            csv_cols = mapping[col]
            if not isinstance(csv_cols, list):
                csv_cols = [csv_cols]

            if len(csv_cols) > 1:
                exprs = []
                for csv_col in csv_cols:
                    rules = filters_by_csv.get(csv_col, {})
                    exprs.append(sql_expr_for(csv_col, rules))
                select_items.append(f"    concat_ws(' ', {', '.join(exprs)}) AS `{col}`")
            else:
                csv_col = csv_cols[0]
                rules = filters_by_csv.get(csv_col, {})

                if not rules or rules == {"trim": True}:
                    select_items.append(f"    `{csv_col}`")
                else:
                    select_items.append(f"    {sql_expr_for(csv_col, rules)} AS `{col}`")

    sql = f"""-- Сгенерировано автоматически, перед вставкой перепроверить!
-- Источник: {staging_db}.{staging_table}
-- Целевая таблица: {target_db}.{target_table}

INSERT INTO {target_db}.{target_table} (
{',\n'.join([f'    {col}' for col in target_cols])}
)
SELECT
{',\n'.join(select_items)}
FROM {staging_db}.{staging_table};
"""

    logger.debug(f"Сгенерированный SQL:\n{sql}")
    return sql
