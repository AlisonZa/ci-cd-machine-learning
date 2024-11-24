from src.utils import logger_obj, e_mail_obj

text = """
Hello
"""


# Entrypoint
if __name__ == "__main__":
    logger_obj.info('Logging succesfully implemented')
    e_mail_obj.send_email(subject= ' ',body= text)
