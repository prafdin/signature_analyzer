import xlsxwriter
from xlsxwriter import utility


class ExcelWorkbook:
    _worksheets = []
    _current_worksheet = 0

    def __init__(self, name):
        self._workbook = xlsxwriter.Workbook(name)
        self._worksheets.append(self._workbook.add_worksheet())
        self._default_format = self._workbook.add_format()

    def write(self, symbol_num, row_num, value, color=None):
        cell_format = self._default_format

        if color is not None:
            cell_format = self._workbook.add_format({'bg_color': color})

        self._worksheets[self._current_worksheet].write(
            utility.xl_col_to_name(symbol_num) + str(row_num),
            value,
            cell_format
        )

    def dump(self):
        self._workbook.close()

    def change_worksheet(self, ind):
        if len(ind >= self._worksheets):
            raise Exception(f"There is not worksheet with {ind} index")
        _current_worksheet = ind
