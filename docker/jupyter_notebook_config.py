# coding=utf-8
import os

from IPython.lib import passwd  # pylint: disable=no-name-in-module

c = get_config()  # type: ignore # pylint: disable=undefined-variable # noqa
c.NotebookApp.ip = "0.0.0.0"
c.NotebookApp.port = int(os.getenv("PORT", 8888))
c.NotebookApp.open_browser = False

password = os.environ.get("PASSWORD", False)
if password:
    c.NotebookApp.password = passwd(password)
else:
    c.NotebookApp.password = ""
    c.NotebookApp.token = ""

try:
    del os.environ["PASSWORD"]
except KeyError:
    pass
