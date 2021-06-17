Contributing
============

Contributions are welcome, and they are greatly appreciated!

You can contribute in many ways:

Types of Contributions
----------------------
Report Bugs
###########
Report bugs by filing an `issue`_.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
########
Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
##################
Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
###################
Dlup could always use more documentation, whether as part of the
official dlup docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
###############
The best way to send feedback is to file an `issue`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
############
Ready to contribute? Here's how to set up `dlup` for local development.

1. Fork the `dlup` repo on GitHub.
2. Clone your fork locally:

.. code-block:: console

    $ git clone git@github.com:your_name_here/dlup.git

3. Create a virtual environment either through conda, docker or virtualenv

4. Install git pre-commit hooks

   - Install `pre-commit`_.
   - Install pre-commit hooks: :code:`pre-commit install`.

5. Install your local copy into a virtual environment:

.. code-block:: console

    $ cd dlup/
    $ pip install --editable ".[dev]"

4. Create a branch for local development:

.. code-block:: console

    $ git checkout -b name-of-your-bugfix-or-feature`

Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests and the pre-commit hooks and the
   tests, including testing other Python versions with pre-commit:

.. code-block:: console

   $ make tests
   $ tox
   $ pre-commit

To get pylint and tox, just pip install them.

6. Commit your changes and push your branch to GitHub:

.. code-block:: console

   $ git add .
   $ git commit -m "Your detailed description of your changes."
   $ git push origin name-of-your-bugfix-or-feature


7. Submit a pull request through the GitHub website.

Pull Request Guidelines
#######################
Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.8 and 3.9, and for PyPy. Check
   https://travis-ci.com/NKI-AI/dlup/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
####
To run a subset of tests:

.. code-block:: console

    $ pytest tests.test_dlup`

Deploying
#########
A reminder for the maintainers on how to deploy.
Make sure all your changes are committed. Then run:

.. code-block:: console

    $ bump2version patch # possible: major / minor / patch
    $ git push


.. _pre-commit: https://pre-commit.com/
.. _GitHub repository: https://github.com/NKI-AI/dlup
.. _issue: https://github.com/NKI-AI/issues
