import sys


def error_message_detail(error,error_detail:sys):
    # sourcery skip: inline-immediately-returned-variable
    _,_,exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred, Python Script Name: [{0}] at Line Number: [{1}], Error Message: [{2}], Details Of Error Detection: [{3}], Location: [{4}]".format(
        file_name, exc_tb.tb_lineno, str(error), error_detail, exc_tb
    )

    return error_message





class CustomException(Exception):
    def __init__(self,error_message,error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail = error_detail
        )
    
    def __str__(self):
        return self.error_message