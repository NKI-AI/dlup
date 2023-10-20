import re
from datetime import datetime
from typing import Callable, Dict


def citation_regex(new_date, content):
    """
    Replaces the date in the CITATION.cff file with the new date
    """
    return re.sub(r"date-released: \d{4}-\d{2}-\d{2}", f"date-released: {new_date}", content)


readme_regexes = {
    "bibtex_year": lambda year, content: re.sub(r"\{\d{4}\}", f"{{{year}}}", content),  # {2023} in bibtex
    "bibtex_month": lambda month, content: re.sub(r"\{\d{1,2}\}", f"{{{month}}}", content),  # {5} or {12} in bibtex
    "plain_bib_year": lambda year, content: re.sub(r"(\d{4})", str(year), content),  # (2023) in the plain bibliography
}


def bump_citation_file(
    new_date: str,
    filename: str = "CITATION.cff",
    regex_function: Callable = citation_regex,
) -> None:
    """
    Runs a specific regex function on the defined citation file
    """
    with open(filename, "r") as f:
        content = f.read()

    with open(filename, "w") as f:
        content = regex_function(new_date, content)
        f.write(content)


def bump_readme_file(
    year: str,
    month: str,
    regex_functions: Dict[str, Callable] = readme_regexes,
    filename: str = "README.md",
) -> None:
    """
    Runs the required regex functions on the README file
    """
    with open(filename, "r") as f:
        content = f.read()

    with open(filename, "w") as f:
        content = regex_functions["bibtex_year"](year, content)
        content = regex_functions["bibtex_month"](month, content)
        content = regex_functions["plain_bib_year"](year, content)
        f.write(content)


def bump_date() -> None:
    """
    A developer function to bump the date when a version is bumped.

    Run after bump2version is run, since bump2version doesn't allow updating
    the date in e.g. README.md and CITATION.cff.
    """
    new_date = datetime.now().strftime("%Y-%m-%d")
    year = str(datetime.now().year)
    month = str(datetime.now().month)

    bump_citation_file(new_date=new_date)
    bump_readme_file(year=year, month=month)


if __name__ == "__main__":
    bump_date()
