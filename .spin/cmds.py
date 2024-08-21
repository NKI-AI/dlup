import subprocess
import webbrowser
from pathlib import Path

import click


@click.group()
def cli():
    """DLUP development commands"""
    pass


@cli.command()
def build():
    """🔧 Build the project"""
    subprocess.run(["meson", "setup", "builddir", "--prefix", str(Path.cwd())], check=True)
    subprocess.run(["meson", "compile", "-C", "builddir"], check=True)
    subprocess.run(["meson", "install", "-C", "builddir"], check=True)


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.argument("tests", nargs=-1)
def test(verbose, tests):
    """🔍 Run tests"""
    cmd = ["pytest"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=dlup --cov=tests --cov-report=html --cov-report=term"])
    if tests:
        cmd.extend(tests)
    subprocess.run(cmd, check=True)


@cli.command()
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.argument("tests", nargs=-1)
def coverage(verbose, tests):
    """🧪 Run tests and generate coverage report"""
    cmd = ["pytest", "--cov=dlup", "--cov=tests", "--cov-report=html", "--cov-report=term"]
    if verbose:
        cmd.append("-v")
    if tests:
        cmd.extend(tests)
    subprocess.run(cmd, check=True)
    coverage_path = Path.cwd() / "htmlcov" / "index.html"
    webbrowser.open(f"file://{coverage_path.resolve()}")


@cli.command()
def mypy():
    """🦆 Run mypy for type checking"""
    subprocess.run(["mypy", "dlup"], check=True)


@cli.command()
def lint():
    """🧹 Run linting"""
    subprocess.run(["flake8", "dlup", "tests"], check=True)


@cli.command()
def ipython():
    """💻 Start IPython"""
    subprocess.run(["ipython"], check=True)


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def python(args):
    """🐍 Start Python"""
    subprocess.run(["python"] + list(args), check=True)


@cli.command()
def docs():
    """📚 Build documentation"""
    docs_dir = Path("docs")
    build_dir = docs_dir / "_build"

    # Remove old builds
    if build_dir.exists():
        for item in build_dir.iterdir():
            if item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_file():
                        subitem.unlink()
                item.rmdir()
            else:
                item.unlink()

    # Generate API docs
    subprocess.run(["sphinx-apidoc", "-o", str(docs_dir), "dlup"], check=True)

    # Build HTML docs
    subprocess.run(["sphinx-build", "-b", "html", str(docs_dir), str(build_dir / "html")], check=True)


@cli.command()
def viewdocs():
    """📖 View documentation in browser"""
    doc_path = Path.cwd() / "docs" / "_build" / "html" / "index.html"
    webbrowser.open(f"file://{doc_path.resolve()}")


@cli.command()
def uploaddocs():
    """📤 Upload documentation"""
    docs()
    source = Path.cwd() / "docs" / "_build" / "html"
    subprocess.run(
        ["rsync", "-avh", f"{source}/", "docs@aiforoncology.nl:/var/www/html/docs/dlup", "--delete"], check=True
    )


@cli.command()
def servedocs():
    """🖥️ Serve documentation and watch for changes"""
    subprocess.run(["sphinx-autobuild", "docs", "docs/_build/html"], check=True)


@cli.command()
def clean():
    """🧹 Clean all build, test, coverage, docs and Python artifacts"""
    dirs_to_remove = ["build", "dist", "_skbuild", ".eggs", "htmlcov", ".tox", ".pytest_cache", "docs/_build"]
    for dir in dirs_to_remove:
        path = Path(dir)
        if path.exists():
            for item in path.glob("**/*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            path.rmdir()

    patterns_to_remove = ["*.egg-info", "*.egg", "*.pyc", "*.pyo", "*~", "__pycache__", "*.o", "*.so"]
    for pattern in patterns_to_remove:
        for path in Path(".").rglob(pattern):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    cython_compiled_files = ["dlup/_background.c"]
    for file in cython_compiled_files:
        path = Path(file)
        if path.exists():
            path.unlink()

@cli.command()
def release():
    """📦 Package and upload a release"""
    dist()
    subprocess.run(["twine", "upload", "dist/*"], check=True)


@cli.command()
def changelog():
    return


@cli.command()
def dist():
    """📦 Build source and wheel package"""
    clean()
    subprocess.run(["python", "-m", "build"], check=True)
    subprocess.run(["ls", "-l", "dist"], check=True)


if __name__ == "__main__":
    cli()
