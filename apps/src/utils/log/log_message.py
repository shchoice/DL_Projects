import traceback


class LogMessage:
    def get_line_number(self, exc_traceback):
        return exc_traceback.tb_lineno

    def stack_trace(self, exc_type, exc_value, exc_traceback):
        return "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    def make_log_message(self, line_no, stack_trace):
        return f'Error at line {line_no}: {stack_trace}'
