.. development:

Development
===========

If you're interested in contributing to Silicone, we'd love to have you on board!
This section of the docs will (once we've written it) detail how to get setup to contribute and how best to communicate.

.. contents:: :local:

Contributing
------------

Guildelines re contributing


Getting setup
-------------

How to get setup


Formatting
----------

To help us focus on what the code does, not how it looks, we use a couple of automatic formatting tools.
These automatically format the code for us and tell use where the errors are.
To use them, after setting yourself up (see `Getting setup`_), simply run ``make format``.
Note that ``make format`` can only be run if you have committed all your work i.e. your working directory is 'clean'.
This restriction is made to ensure that you don't format code without being able to undo it, just in case something goes wrong.


Buiding the docs
----------------

After setting yourself up (see `Getting setup`_), building the docs is as simple as running ``make docs`` (note, run ``make -B docs`` to force the docs to rebuild and ignore make when it says '... index.html is up to date').
This will build the docs for you.
You can preview them by opening ``docs/build/html/index.html`` in a browser.

For documentation we use Sphinx_.
To get ourselves started with Sphinx, we started with `this example <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ then used `Sphinx's getting started guide <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.


Gotchas
~~~~~~~

To get Sphinx to generate pdfs (rarely worth the hassle), you require `Latexmk <https://mg.readthedocs.io/latexmk.html>`_.
On a Mac this can be installed with ``sudo tlmgr install latexmk``.
You will most likely also need to install some other packages (if you don't have the full distribution).
You can check which package contains any missing files with ``tlmgr search --global --file [filename]``.
You can then install the packages with ``sudo tlmgr install [package]``.


Docstring style
~~~~~~~~~~~~~~~

For our docstrings we use numpy style docstrings.
For more information on these, `here is the full guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and `the quick reference we also use <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.


Releasing
---------

Instructions on how to release versions of silicone will go here.


Why is there a ``Makefile`` in a pure Python repository?
--------------------------------------------------------

Whilst it may not be standard practice, a ``Makefile`` is a simple way to automate general setup (environment setup in particular).
Hence we have one here which basically acts as a notes file for how to do all those little jobs which we often forget e.g. setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxillary bits and pieces.

.. _Sphinx: http://www.sphinx-doc.org/en/master/
