Changelog
=========

[v1.3.0] - 27 Apr 2025
----------------------

changed
~~~~~~
- (`#151 <https://github.com/GranthamImperial/silicone/pull/151>`_) Improved error handling and structured logging in ratio-based crunchers and utilities.
- (`#153 <https://github.com/GranthamImperial/silicone/pull/153>`_) Use :func:`pandas.DataFrame.map` in preference to :func:`pandas.DataFrame.applymap` as the later was deprecated in Pandas v2.1.0.

[v1.3.0] - 14 Oct 2022
----------------------

Added
~~~~~
- (`#146 <https://github.com/GranthamImperial/silicone/pull/146>`_) Added the ability to do non-linear interpolation by introducing :class:`Interpolation`.

Changed
~~~~~~~
- (`#149 <https://github.com/GranthamImperial/silicone/pull/149>`_) Checked consistency of time extender database before extending at constant quantiles.
- (`#142 <https://github.com/GranthamImperial/silicone/pull/142>`_) Sped up RMS closest
- (`#146 <https://github.com/GranthamImperial/silicone/pull/146>`_) Deprecated the linear interpolator :class:`LinearInterpolation` in favour of a generic interpolator :class:`Interpolation`.
- (`#150 <https://github.com/GranthamImperial/silicone/pull/150>`_) Added more info when returning an error message in multiple infillers.

Fixed
~~~~~
- (`#149 <https://github.com/GranthamImperial/silicone/pull/149>`_) Ensured RMS closest works with the latest version of pyam. Bugfix for a warning in infill_composite_values
- (`#144 <https://github.com/GranthamImperial/silicone/pull/144>`_) RMS closest no longer causes :class:`pd.core.common.SettingWithCopyWarning` to be raised
- (`#147 <https://github.com/GranthamImperial/silicone/pull/147>`_) Filter prevents including data from the wrong regions in :class:`DecomposeCollectionTimeDepRatio`. Notebook fixed to run with updates in python 3.8.

[v1.2.0] - 28 Sept 2021
-----------------------

Added
~~~~~
- (`#139 <https://github.com/GranthamImperial/silicone/pull/139>`_) Support for pyam-iamc>1.0
- (`#135 <https://github.com/GranthamImperial/silicone/pull/135>`_) Added html documentation of the time projectors

Changed
~~~~~~~
- (`#138 <https://github.com/GranthamImperial/silicone/pull/138>`_) Remove Python3.6 support
- (`#138 <https://github.com/GranthamImperial/silicone/pull/138>`_) Improved speed of :func:`silicone.multiple_infillers.infill_all_required_emissions_for_openscm` by removing multiple loops (note that API did not change)


[v1.1.0] - 12 July 2021
-----------------------

Added
~~~~~
- (`#134 <https://github.com/GranthamImperial/silicone/pull/134>`_) Added Gaurav to author list.
- (`#132 <https://github.com/GranthamImperial/silicone/pull/132>`_) Added an additional time projector (Linear Extender) that simply extends the latest data to reach a specified point or by a constant gradient.
- (`#129 <https://github.com/GranthamImperial/silicone/pull/129>`_) Added an additional time projector (Extend RMS closest) that extends a pathway to cover later times by selecting future data from the closest pathway.
- (`#126 <https://github.com/GranthamImperial/silicone/pull/126>`_) Added the first time projector (Extend latest time quantile) that extends a pathway to cover later times, assuming it remains at the same quantile.

Changed
~~~~~~~
- (`#133 <https://github.com/GranthamImperial/silicone/pull/133>`_) More fixes to allow compatibility with pyam updates.
- (`#131 <https://github.com/GranthamImperial/silicone/pull/131>`_) Updated to allow compatibility with latest versions of pyam, openscm-units, coverage, pytest and black

Fixed
~~~~~
- (`#130 <https://github.com/GranthamImperial/silicone/pull/130>`_) Reformatted files to make the linter happy (no functional changes).


[v1.0.3]
--------

Changed
~~~~~~~
- (`#124 <https://github.com/GranthamImperial/silicone/pull/124>`_) Neatened up the changelog


[v1.0.2] - 4 Jan 2021
---------------------

Changed
~~~~~~~
- (`#121 <https://github.com/GranthamImperial/silicone/pull/121>`_) Updated to openscm-units>0.2

Fixed
~~~~~
- (`#123 <https://github.com/GranthamImperial/silicone/pull/123>`_) Made the installation runner avoid prerelease.


[v1.0.1] - 27 Oct 2020
----------------------

Added
~~~~~
- (`#115 <https://github.com/GranthamImperial/silicone/pull/115>`_) Enabled multiple lead gases to be used with RMS closest cruncher.

Changed
~~~~~~~
- (`#119 <https://github.com/GranthamImperial/silicone/pull/119>`_) Updated to work with pyam v0.8


[v1.0.0] - 9 Sept 2020
----------------------

Initial release
~~~~~~~~~~~~~~~
- (`#116 <https://github.com/GranthamImperial/silicone/pull/116>`_) Pinned black
- (`#113 <https://github.com/GranthamImperial/silicone/pull/113>`_) Added a warning for using ratio-based crunchers with negative values. Fixed some unit conversion todos (not user-facing).
- (`#112 <https://github.com/GranthamImperial/silicone/pull/112>`_) Enabled more general unit conversion, bug fix and improvement for infill_composite_values.
- (`#111 <https://github.com/GranthamImperial/silicone/pull/111>`_) Minor improvements to error messages and documentation.
- (`#110 <https://github.com/GranthamImperial/silicone/pull/110>`_) Gave an option to time_dep_ratio and decompose_collection to ignore model/scenario combinations missing values at some required times.
- (`#108 <https://github.com/GranthamImperial/silicone/pull/108>`_) Added a multiple infiller to split up an aggregate with a remainder. Disabled test for downloading database.
- (`#103 <https://github.com/GranthamImperial/silicone/pull/103>`_) Update github address to GranthamImperial.
- (`#101 <https://github.com/GranthamImperial/silicone/pull/101>`_) Update release docs
- (`#93 <https://github.com/GranthamImperial/silicone/pull/93>`_) Add regular test of install from PyPI
- (`#102 <https://github.com/GranthamImperial/silicone/pull/102>`_) Minor bugfix for nan handling in Equal Quantile Walk.
- (`#100 <https://github.com/GranthamImperial/silicone/pull/100>`_) Added funding info to readme and removed unnecessary files.
- (`#97 <https://github.com/GranthamImperial/silicone/pull/97>`_) Added sections to documentation file so that newer crunchers and multiple infillers are included.
- (`#95 <https://github.com/GranthamImperial/silicone/pull/95>`_) Added sections to notebooks covering all the recent changes.
- (`#94 <https://github.com/GranthamImperial/silicone/pull/94>`_) Added :class:`EqualQuantileWalk`, a cruncher which finds the quantile of the lead variable in the infiller database and returns the same quantile of the follow variable.
- (`#87 <https://github.com/GranthamImperial/silicone/pull/87>`_) Added :class:`TimeDepQuantileRollingWindows`, a cruncher which allows the user to crunch different quantiles in different years.
- (`#86 <https://github.com/GranthamImperial/silicone/pull/86>`_) Slightly changed the definition of quantile rolling windows to make it symmetric (not rounding down).
- (`#83 <https://github.com/GranthamImperial/silicone/pull/83>`_) Added tests for appending results of crunching to the input.
- (`#82 <https://github.com/GranthamImperial/silicone/pull/82>`_) Updated to a later version of pyam and solved todos associated with this. Also added a ``kwargs`` argument to ``infill_all_required``.
- (`#80 <https://github.com/GranthamImperial/silicone/pull/80>`_) Changed the names of crunchers for brevity. Also changed ``lead_gas`` to ``latest_time_ratio`` and included it in ratio notebook.
- (`#78 <https://github.com/GranthamImperial/silicone/pull/78>`_) Changed how quantile rolling windows works by adding an extra interpolate step for smoothness
- (`#77 <https://github.com/GranthamImperial/silicone/pull/77>`_) Added calculation of variance of rank correlation to stats
- (`#76 <https://github.com/GranthamImperial/silicone/pull/76>`_) Removed command-line interface
- (`#75 <https://github.com/GranthamImperial/silicone/pull/75>`_) Updated README
- (`#72 <https://github.com/GranthamImperial/silicone/pull/72>`_) Altered infill_composite_value to allow multiplication by a factor before summing. Removed unnecessary notebooks.
- (`#69 <https://github.com/GranthamImperial/silicone/pull/69>`_) Fixed bug so that ``DatabaseCruncherRMSClosest`` no longer selects scenarios which don't have follower data
- (`#68 <https://github.com/GranthamImperial/silicone/pull/68>`_) More investigatory tools and scripts for calculating and outputting emissions correlations.
- (`#67 <https://github.com/GranthamImperial/silicone/pull/67>`_) Introduce investigatory tools for plotting relations between emissions.
- (`#66 <https://github.com/GranthamImperial/silicone/pull/66>`_) Remove ``Input`` folder in favour of using ``openscm-units``
- (`#65 <https://github.com/GranthamImperial/silicone/pull/65>`_) Add ``format-notebooks`` target to the ``Makefile``
- (`#64 <https://github.com/GranthamImperial/silicone/pull/64>`_) Add basic linters to CI
- (`#61 <https://github.com/GranthamImperial/silicone/pull/61>`_) Switch to using GitHub actions for CI
- (`#60 <https://github.com/GranthamImperial/silicone/pull/60>`_) Update installation docs to reference pip and conda
- (`#62 <https://github.com/GranthamImperial/silicone/pull/62>`_) Minor changes to remove warning messages and remove some todos.
- (`#52 <https://github.com/GranthamImperial/silicone/pull/52>`_) Made the Lead Gas infiller use the average latest data rather than being restricted to a single value. Updated infill_composite_values to work with the latest data.
- (`#51 <https://github.com/GranthamImperial/silicone/pull/51>`_) Split the notebooks into chapters with minor changes to the text. Moved a script function into utilities to download data.
- (`#49 <https://github.com/GranthamImperial/silicone/pull/49>`_) Rewrote the documentation and notebooks to update, split up information and clarify.
- (`#48 <https://github.com/GranthamImperial/silicone/pull/48>`_) Introduced multiple_infiller function to calculate the composite values from the constituents.
- (`#47 <https://github.com/GranthamImperial/silicone/pull/47>`_) Made an option for quantile_rolling_windows to infill using the ratio of lead to follow data.
- (`#46 <https://github.com/GranthamImperial/silicone/pull/46>`_) Made the time-dependent ratio infiller only use data where the leader has the same sign.
- (`#45 <https://github.com/GranthamImperial/silicone/pull/45>`_) Made infill_all_required_emissions_for_openscm, the second multiple-infiller function.
- (`#44 <https://github.com/GranthamImperial/silicone/pull/44>`_) Made decompose_collection_with_time_dep_ratio, the first multiple-infiller function.
- (`#43 <https://github.com/GranthamImperial/silicone/pull/43>`_) Implemented new util functions for downloading data, unit conversion and data checking.
- (`#41 <https://github.com/GranthamImperial/silicone/pull/41>`_) Added a cruncher to interpolate values between data from specific scenarios. Only test notebooks with lax option.
- (`#32 <https://github.com/GranthamImperial/silicone/pull/32>`_) Raise `ValueError` when asking to infill a case with no data
- (`#27 <https://github.com/GranthamImperial/silicone/pull/27>`_) Developed the constant ratio cruncher
- (`#21 <https://github.com/GranthamImperial/silicone/pull/21>`_) Developed the time-dependent ratio cruncher
- (`#20 <https://github.com/GranthamImperial/silicone/pull/20>`_) Clean up the quantiles cruncher and test rigorously
- (`#19 <https://github.com/GranthamImperial/silicone/pull/19>`_) Add releasing docs plus command-line entry point tests
- (`#14 <https://github.com/GranthamImperial/silicone/pull/14>`_) Add root-mean square closest pathway cruncher
- (`#13 <https://github.com/GranthamImperial/silicone/pull/13>`_) Get initial work (see `#11 <https://github.com/GranthamImperial/silicone/pull/11>`_) into package structure, still requires tests (see `#16 <https://github.com/GranthamImperial/silicone/pull/16>`_)
- (`#12 <https://github.com/GranthamImperial/silicone/pull/12>`_) Add BSD-3-Clause license
- (`#9 <https://github.com/GranthamImperial/silicone/pull/9>`_) Add lead gas cruncher
- (`#6 <https://github.com/GranthamImperial/silicone/pull/6>`_) Update development docs
- (`#5 <https://github.com/GranthamImperial/silicone/pull/5>`_) Put notebooks under CI
- (`#4 <https://github.com/GranthamImperial/silicone/pull/4>`_) Add basic documentation structure
- (`#1 <https://github.com/GranthamImperial/silicone/pull/1>`_) Added pull request and issues templates
