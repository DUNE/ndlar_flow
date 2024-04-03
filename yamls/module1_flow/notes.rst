============
module1_flow
============

Resources
=========
* ``yamls/module1_flow/resources/RunData.yaml``

  Copied and modified from ``yaml/proto_nd_flow/resources/RunData.yaml``

* ``yamls/module1_flow/resources/Geometry.yaml``

  Copied and modified from ``yamls/proto_nd_flow/resources/Geometry.yaml`` I set ``det_geometry_file`` to ``data/module1_flow/module0.yaml``. I don't know if that's alright, and I don't think module0 accepted that option. ``crs_geometry_file`` is set to what I found in previously flowed file's metadata. Angela provided me with a  ``lrs_geometry_file``. Also, some module workflows had ``network_agnostic`` True and others False; not sure which is correct. 

* ``yamls/module1_flow/resources/LArData.yaml``

  Copied and modified from ``yaml/proto_nd_flow/resources/LArData.yaml``.

  Previously flowed files had an ``electron_lifetime`` of 900 us.

  ``module0_flow`` has ``electron_lifetime_file`` which is not in ``proto_nd_flow``. It looks like modules 2 and 3 workflows set the lifetime to 2600 us. module0 workflow gave both a lifetime file and an electron_lifetime. module0 also gives a vdrift mode and LAr temperature. 


Workflows
=========

Charge
------
1. ``yamls/module1_flow/workflows/charge/charge_event_building.yaml``

   Originally copied and modified from ``ndlar_flow/yamls/proto_nd_flow/workflows/charge/charge_event_building.yaml``. Only difference is that the ``.yaml`` files now point to ``module1_flow`` specific files.

* ``yamls/module1_flow/reco/charge/RawEventGenerator.yaml``

  Copied and modified from ``yamls/proto_nd_flow/reco/charge/RawEventGenerator.yaml``.

  Removed ``mc_tracks_dset_name``.

  Set ``nhit_cuts`` to 5.

2. ``yamls/module1_flow/workflows/charge/charge_event_reconstruction.yaml``

   Copied and modified from ``ndlar_flow/yamls/proto_nd_flow/workflows/charge/charge_event_reconstruction.yaml``

* ``yamls/module1_flow/reco/charge/TimestampCorrector.yaml``

  Copied and modified from ``yamls/proto_nd_flow/reco/charge/TimestampCorrector.yaml``
  Using numbers I found in previously flowed file's metadata.

* ``yamls/module1_flow/reco/charge/ExternalTriggerFinder.yaml``

  Copied from ``yamls/proto_nd_flow/reco/charge/ExternalTriggerFinder.yaml``. No changes made .

* ``yamls/module1_flow/reco/charge/RawHitBuilder.yaml``

  Copied and modified from ``yamls/proto_nd_flow/reco/charge/RawHitBuilder.yaml``

  Included ``configuration_file`` and ``pedestal_file``, and set them to what I found in previously flowed file's metadata. I think this was renamed from ``yamls/module0_flow/reco/charge/HitBuilder.yaml`` in the ``module0_flow``. In module workflows 2 and 3, the option ``network_agnostic: True`` is set. Not sure if we want this here too. 

* ``yamls/module1_flow/reco/charge/EventBuilder.yaml``

  Copied from ``yamls/proto_nd_flow/reco/charge/EventBuilder.yaml``. No changes.


3. ``yamls/module1_flow/workflows/combined/combined_reconstruction.yaml``

   Copied and modified from ``yamls/proto_nd_flow/workflows/combined/combined_reconstruction.yaml``. Only difference is that the ``.yaml`` files point to ``module1_flow`` specific files. ``proto_nd_flow`` only had a ``t0_reco`` step, while module[0,2,3] workflows have ``drift_reco``, ``electron_lifetime_corr``, ``tracklet_reco``, and module[0,2] workflows have ``tracklet_merge``.

* ``yamls/module1_flow/reco/combined/T0Reconstruction.yaml``

  Copied from ``yamls/proto_nd_flow/reco/combined/T0Reconstruction.yaml``. No changes made. Has an extra parameter compared to module[0,2,3] workflows called ``raw_hits_dset_name: 'charge/raw_hits'``.

4. ``yamls/module1_flow/workflows/charge/prompt_calibration.yaml``

   Copied and modified from yamls/proto_nd_flow/workflows/charge/prompt_calibration.yaml. Only difference is that the ``.yaml`` files point to ``module1_flow`` specific files. I don't see a corresponding file for modules[0,2,3] workflows.

* ``yamls/module1_flow/reco/charge/CalibHitBuilder.yaml``

  Copied and modified from ``yamls/proto_nd_flow/reco/charge/CalibHitBuilder.yaml``.

  Added option for ``pedestal_file`` and ``configuration_file``, using inputs found in previoulsy flowed file metadata. I don't see this file for module[0,2,3] workflows.

5. ``yamls/module1_flow/workflows/charge/final_calibration.yaml``

   Copied and modified from ``yamls/proto_nd_flow/workflows/charge/final_calibration.yaml``. Only difference is that ``.yaml`` files now point to ``module1_flow`` specific files. Don't see corresponding file for module[0,2,3] workflows.

* ``yamls/module1_flow/reco/charge/CalibHitMerger.yaml``

  Copied and modified from ``yamls/proto_nd_flow/reco/charge/CalibHitMerger.yaml``. Maybe corresponds to ``yamls/module0_flow/reco/charge/HitMerger.yaml`` in ``module0``? Doesn't exist for module[2,3] workflows.

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

  Copied and modified from ``yamls/proto_nd_flow/reco/light/WaveformCalib.yaml``. For ``gain``, I created an input file using gain corrections Livio sent me in ``mod1_gain_corrected.csv``. The code to make the gains is found in ``gains_and_thresholds.ipynb   ``. 

* ``yamls/module1_flow/reco/light/WaveformSum.yaml``

  Copied from ``yamls/proto_nd_flow/reco/light/WaveformSum.yaml``. Other module workflows have ``gain`` and ``gain_mc`` parameters. 

* ``yamls/module1_flow/reco/light/SiPMHitFinder.yaml``

  Copied from ``yamls/proto_nd_flow/reco/light/SiPMHitFinder.yaml``. ``near_sample`` parameter is different. Here, ``threshold`` is a single constant, while other module workflows point to a ``siplm_threshold.yaml`` file. 

* ``yamls/module1_flow/reco/light/SumHitFinder.yaml``

  Copied from ``yamls/proto_nd_flow/reco/light/SumHitFinder.yaml``. Does not exist for other module workflows. Is ``threshold`` assuming 8 TPCSs? 
