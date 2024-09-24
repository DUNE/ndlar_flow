============
module1_flow
============

Resources
=========
* ``yamls/module2_flow/resources/RunData.yaml``

  Copied and modified from ``yaml/module1_flow/resources/RunData.yaml`` I made my own runlist.txt, I'll have to double check it.

* ``yamls/module2_flow/resources/Geometry.yaml``

  Copied and modified from module1's. I set ``det_geometry_file``. ``data/module2_flow/module2.yaml`` is copied from module1's, but with the tile layout unswapped (Need to check if it comes out right). ``crs_geometry_file`` is set to ``/global/cfs/projectdirs/dune/www/data/Module2/flow_configs/multi_tile_layout-2022_11_18_04_35_CET.yaml``. ``lrs_geometry_file`` is set to the same as module1's.

* ``yamls/module1_flow/resources/LArData.yaml``

    Copied from module1's.


Workflows
=========

Charge
------
1. ``yamls/module2_flow/workflows/charge/charge_event_building.yaml``

   Originally copied and modified from module1's.

* ``yamls/module2_flow/reco/charge/RawEventGenerator.yaml``

  Same as module1's. 

2. ``yamls/module2_flow/workflows/charge/charge_event_reconstruction.yaml``

   Copied and modified from module1's.

* ``yamls/module2_flow/reco/charge/TimestampCorrector.yaml``

  Copied and modified from module1's.
  Using numbers Karolina derived. 

* ``yamls/module2_flow/reco/charge/ExternalTriggerFinder.yaml``

  Copied from module1's. No changes made.

* ``yamls/module2_flow/reco/charge/RawHitBuilder.yaml``

  Copied and modified from ``yamls/proto_nd_flow/reco/charge/RawHitBuilder.yaml``

  Included ``configuration_file`` and ``pedestal_file``, and set them to what I found in previously flowed file's metadata. I think this was renamed from ``yamls/module0_flow/reco/charge/HitBuilder.yaml`` in the ``module0_flow``. In module workflows 2 and 3, the option ``network_agnostic: True`` is set. Not sure if we want this here too. 

* ``yamls/module2_flow/reco/charge/EventBuilder.yaml``

  Copied from module1's.

3. ``yamls/module2_flow/workflows/combined/combined_reconstruction.yaml``

  Copied from module1's. No changes made.

* ``yamls/module2_flow/reco/combined/T0Reconstruction.yaml``

  Copied from module1's. No changes made.

4. ``yamls/module1_flow/workflows/charge/prompt_calibration.yaml``

    Copied and moified from module1's. 

* ``yamls/module1_flow/reco/charge/CalibHitBuilder.yaml``

  Copied and modified from module1's. Used ``pedestal_file`` and ``configuration_file`` found in older module2 yaml files.

5. ``yamls/module1_flow/workflows/charge/final_calibration.yaml``

    Copied and moified from module1's. 

* ``yamls/module1_flow/reco/charge/CalibHitMerger.yaml``

    Copied and moified from module1's. 

  Removed ``mc_hit_frac_dset_name``.

    Light
    -----
    1. ``yamls/module1_flow/workflows/light/light_event_building_adc64.yaml``

       Copied and modified from ``yamls/module3_flow/workflows/light/light_event_building_adc64.yaml``. The equivalent file did not exist in ``proto_nd_flow``. Only difference is that the ``.yaml`` files now point to a ``module1_flow`` specific file.

    * ``yamls/module1_flow/reco/light/LightADC64EventGenerator.yaml``

      Copied and modified from ``yamls/module3_flow/reco/light/LightADC64EventGenerator.yaml``. Set the ``sn_table`` arguments, I need to remember from where.

    2. ``yamls/module1_flow/workflows/light/light_event_reconstruction.yaml``

       Copied and modified from ``yamls/proto_nd_flow/workflows/light/light_event_reconstruction.yaml``. Only difference is that the ``.yaml`` files now point to a ``module1_flow`` specific file. Compared to module 0 workflow, there are three extra steps: ``wvfm_calib``, ``sipm_hit_finder``, ``sum_hit_finder``.

    * ``yamls/module1_flow/reco/light/LightTimestampCorrector.yaml``

      Copied and modified from ``yamls/proto_nd_flow/reco/light/LightTimestampCorrector.yaml``. Changed ``slope`` to only have two TPC values. I noticed that all other modules have slopes (0: -1.18e-7, 1: 1.18e-7), while I kept them set to 0. Not sure what module1 wants. 

    * ``yamls/module1_flow/reco/light/WaveformNoiseFilter.yaml``

      Copied from ``yamls/proto_nd_flow/reco/light/WaveformNoiseFilter.yaml``. Option ``filter_channels`` differs from others modules.

    * ``yamls/module1_flow/reco/light/WaveformDeconvolution.yaml``

      Copied and modified from ``yamls/proto_nd_flow/reco/light/WaveformDeconvolution.yaml``.
      ``noise_spectrum_filename``, ``signal_spectrum_filename``, ``signal_impulse_filename`` were set generated using ``run_light_extract_response.sh``, with ``0cd913fb_20220211_074023.data`` as the input file.
      Option ``filter_channels`` differs from other modules.

    * ``yamls/module1_flow/reco/light/WaveformAlign.yaml``

      Copied from ``yamls/proto_nd_flow/reco/light/WaveformAlign.yaml``. Is ``sim_latency`` a simulation parameter that should be removed? Other module workflows have ``busy_channel: All: 0`` parameter. 

    * ``yamls/module1_flow/reco/light/WaveformCalib.yaml``

      Copied and modified from ``yamls/proto_nd_flow/reco/light/WaveformCalib.yaml``. For ``gain``, I created an input file using gain corrections Livio sent me in ``mod1_gain_corrected.csv``. The code to make the gains is found in ``gains_and_thresholds.ipynb``. 

    * ``yamls/module1_flow/reco/light/WaveformSum.yaml``

      Copied from ``yamls/proto_nd_flow/reco/light/WaveformSum.yaml``. Other module workflows have ``gain`` and ``gain_mc`` parameters. 

    * ``yamls/module1_flow/reco/light/SiPMHitFinder.yaml``

      Copied and modified from ``yamls/proto_nd_flow/reco/light/SiPMHitFinder.yaml``. ``near_sample`` parameter is different. I generated a ``sipm_threshold.yaml`` file using ``gains_and_thresholds.ipynb``. 

    * ``yamls/module1_flow/reco/light/SumHitFinder.yaml``

      Copied and modified from ``yamls/proto_nd_flow/reco/light/SumHitFinder.yaml``. I generated a ``sum_threshold.yaml`` threshold file using ``gains_and_thresholds.ipynb``. 
