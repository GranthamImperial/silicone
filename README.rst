Silicone
========

+--------+--------+-----------+
| Basics | |Docs| | |License| |
+--------+--------+-----------+

+-------------------+----------------+-----------+
| Repository health | |Build Status| | |Codecov| |
+-------------------+----------------+-----------+

+-----------------+------------------+
| Latest releases | |Latest Version| |
+-----------------+------------------+

+-----------------+----------------+---------------+------------------------------+
| Latest activity | |Contributors| | |Last Commit| | |Commits Since Last Release| |
+-----------------+----------------+---------------+------------------------------+

.. sec-begin-long-description
.. sec-begin-index


Silicone is a Python package which can be used to infer
emissions from other emissions data. It is intended to 'infill' integrated
assessment model (IAM) data so that their scenarios quantify more climate-relevant
emissions than is natively reported by the IAMs themselves.
It does this by comparing the incomplete emissions set to complete data. It uses the
relationships within the complete data to make informed infilling estimates of
otherwise missing emissions timeseries.
For example, it can add emissions of aerosol precurors based on carbon dioxide emissions
and infill nitrous oxide emissions based on methane, or split HFC emissions
pathways into emissions of different specific HFC gases.

.. sec-end-index

License
-------

.. sec-begin-license

Silicone is free software under a BSD 3-Clause License, see
`LICENSE <https://github.com/znicholls/silicone/blob/master/LICENSE>`_.

.. sec-end-license
.. sec-end-long-description

.. sec-begin-installation

Installation
------------

Silicone can be installed with pip

.. code:: bash

    pip install silicone

If you also want to run the example notebooks, install additional
dependencies using

.. code:: bash

    pip install silicone[notebooks]

**Coming soon** Silicone can also be installed with conda

.. code:: bash

    conda install -c conda-forge silicone

.. sec-end-installation

Documentation
-------------

Documentation can be found at our `documentation pages <https://silicone.readthedocs.io/en/latest/>`_
(we are thankful to `Read the Docs <https://readthedocs.org/>`_ for hosting us).

Contributing
------------

Please see the `Development section of the docs <https://silicone.readthedocs.io/en/latest/development.html>`_.

.. sec-begin-links

.. |Docs| image:: https://readthedocs.org/projects/silicone/badge/?version=latest
    :target: https://silicone.readthedocs.io/en/latest/
.. |License| image:: https://img.shields.io/github/license/znicholls/silicone.svg
    :target: https://github.com/znicholls/silicone/blob/master/LICENSE
.. |Build Status| image:: https://travis-ci.com/znicholls/silicone.svg?branch=master
    :target: https://travis-ci.com/znicholls/silicone
.. |Codecov| image:: https://img.shields.io/codecov/c/github/znicholls/silicone.svg
    :target: https://codecov.io/gh/znicholls/silicone/branch/master/graph/badge.svg
.. |Latest Version| image:: https://img.shields.io/github/tag/znicholls/silicone.svg
    :target: https://github.com/znicholls/silicone/releases
.. |Last Commit| image:: https://img.shields.io/github/last-commit/znicholls/silicone.svg
    :target: https://github.com/znicholls/silicone/commits/master
.. |Commits Since Last Release| image:: https://img.shields.io/github/commits-since/znicholls/silicone/latest.svg
    :target: https://github.com/znicholls/silicone/commits/master
.. |Contributors| image:: https://img.shields.io/github/contributors/znicholls/silicone.svg
    :target: https://github.com/znicholls/silicone/graphs/contributors

.. sec-end-links
