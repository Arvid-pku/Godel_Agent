import io
import traceback

def wrap_solver(solver):
    def try_solver(*args, **kwargs):
        try:
            return solver(*args, **kwargs)
        except:
            exception_stringio = io.StringIO()
            traceback.print_exc(file=exception_stringio, limit=5)
            return "Error Message:\n" + exception_stringio.getvalue()
    return try_solver