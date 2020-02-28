Changelog
=========

master
------
- (`#48 <https://github.com/znicholls/silicone/pull/48>`_) Introduced multiple_infiller function to calculate the composite values from the constituents.
- (`#47 <https://github.com/znicholls/silicone/pull/47>`_) Made an option for quantile_rolling_windows to infill using the ratio of lead to follow data.
- (`#46 <https://github.com/znicholls/silicone/pull/46>`_) Made the time-dependent ratio infiller only use data where the leader has the same sign.
- (`#45 <https://github.com/znicholls/silicone/pull/45>`_) Made infill_all_required_emissions_for_openscm, the second multiple-infiller function.
- (`#44 <https://github.com/znicholls/silicone/pull/44>`_) Made decompose_collection_with_time_dep_ratio, the first multiple-infiller function.
- (`#43 <https://github.com/znicholls/silicone/pull/43>`_) Implemented new util functions for downloading data, unit conversion and data checking.
- (`#41 <https://github.com/znicholls/silicone/pull/41>`_) Added a cruncher to interpolate values between data from specific scenarios. Only test notebooks with lax option.
- (`#32 <https://github.com/znicholls/silicone/pull/32>`_) Raise `ValueError` when asking to infill a case with no data
- (`#27 <https://github.com/znicholls/silicone/pull/27>`_) Developed the constant ratio cruncher
- (`#21 <https://github.com/znicholls/silicone/pull/21>`_) Developed the time-dependent ratio cruncher
- (`#20 <https://github.com/znicholls/silicone/pull/20>`_) Clean up the quantiles cruncher and test rigorously
- (`#19 <https://github.com/znicholls/silicone/pull/19>`_) Add releasing docs plus command-line entry point tests
- (`#14 <https://github.com/znicholls/silicone/pull/14>`_) Add root-mean square closest pathway cruncher
- (`#13 <https://github.com/znicholls/silicone/pull/13>`_) Get initial work (see `#11 <https://github.com/znicholls/silicone/pull/11>`_) into package structure, still requires tests (see `#16 <https://github.com/znicholls/silicone/pull/16>`_)
- (`#12 <https://github.com/znicholls/silicone/pull/12>`_) Add BSD-3-Clause license
- (`#9 <https://github.com/znicholls/silicone/pull/9>`_) Add lead gas cruncher
- (`#6 <https://github.com/znicholls/silicone/pull/6>`_) Update development docs
- (`#5 <https://github.com/znicholls/silicone/pull/5>`_) Put notebooks under CI
- (`#4 <https://github.com/znicholls/silicone/pull/4>`_) Add basic documentation structure
- (`#1 <https://github.com/znicholls/silicone/pull/1>`_) Added pull request and issues templates
