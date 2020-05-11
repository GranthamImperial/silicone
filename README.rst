Silicone
========

+--------+-----------+
| Basics | |License| |
+--------+-----------+

+-------------------+----------------+-----------+--------+
| Repository health | |Build Status| | |Codecov| | |Docs| |
+-------------------+----------------+-----------+--------+

+-----------------+----------------+----------------+------------------+
| Latest releases | |PyPI Install| | |PyPI Version| | |Latest Version| |
+-----------------+----------------+----------------+------------------+

+-----------------+----------------+---------------+------------------------------+
| Latest activity | |Contributors| | |Last Commit| | |Commits Since Last Release| |
+-----------------+----------------+---------------+------------------------------+

.. sec-begin-long-description
.. sec-begin-index


Silicone is a Python package which can be used to infer emissions from other emissions data.
It is intended to 'infill' integrated assessment model (IAM) data so that their scenarios
quantify more climate-relevant emissions than are natively reported by the IAMs themselves.
It does this by comparing the incomplete emissions set to complete data from other sources.
It uses the relationships within the complete data to make informed infilling estimates of
otherwise missing emissions timeseries.
For example, it can add emissions of aerosol precurors based on carbon dioxide emissions
and infill nitrous oxide emissions based on methane, or split HFC emissions pathways into
emissions of different specific HFC gases.


.. sec-end-index

License
-------

.. sec-begin-license

Silicone is free software under a BSD 3-Clause License, see
`LICENSE <https://github.com/GranthamImperial/silicone/blob/master/LICENSE>`_.

.. sec-end-license

.. sec-begin-funders

Funders
-------
This project has received funding from the European Union Horizon 2020 research and
innovation programme under grant agreement No 820829.

.. sec-end-funders
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
.. |License| image:: https://img.shields.io/github/license/GranthamImperial/silicone.svg
    :target: https://github.com/GranthamImperial/silicone/blob/master/LICENSE
.. |Build Status| image:: https://github.com/GranthamImperial/silicone/workflows/Silicone%20CI-CD/badge.svg
    :target: https://github.com/GranthamImperial/silicone/actions?query=workflow%3A%22Silicone+CI-CD%22
.. |Codecov| image:: https://img.shields.io/codecov/c/github/GranthamImperial/silicone.svg
    :target: https://codecov.io/gh/GranthamImperial/silicone/branch/master/graph/badge.svg
.. |Latest Version| image:: https://img.shields.io/github/tag/GranthamImperial/silicone.svg
    :target: https://github.com/GranthamImperial/silicone/releases
.. |PyPI Install| image:: https://github.com/GranthamImperial/silicone/workflows/Test%20PyPI%20install/badge.svg
    :target: https://github.com/GranthamImperial/silicone/actions?query=workflow%3A%22Test+PyPI+install%22
.. |PyPI Version| image:: https://img.shields.io/pypi/v/silicone.svg
    :target: https://pypi.org/project/silicone/
.. |Last Commit| image:: https://img.shields.io/github/last-commit/GranthamImperial/silicone.svg
    :target: https://github.com/GranthamImperial/silicone/commits/master
.. |Commits Since Last Release| image:: https://img.shields.io/github/commits-since/GranthamImperial/silicone/latest.svg
    :target: https://github.com/GranthamImperial/silicone/commits/master
.. |Contributors| image:: https://img.shields.io/github/contributors/GranthamImperial/silicone.svg
    :target: https://github.com/GranthamImperial/silicone/graphs/contributors

.. sec-end-links
