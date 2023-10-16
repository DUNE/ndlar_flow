============
module1_flow
============

Resources
=========
* ``yamls/module1_flow/resources/RunData.yaml``

  Copied and modified from ``yaml/proto_nd_flow/resources/RunData.yaml``

* ``yamls/module1_flow/resources/Geometry.yaml``

  Copied and modified from ``yaml/proto_nd_flow/resources/Geometry.yaml``

* ``yamls/module1_flow/resources/LArData.yaml``

  Copied and modified from ``yaml/proto_nd_flow/resources/LArData.yaml``.

  Previously flowed files had an ``electron_lifetime`` of 900 us.

  ``module0_flow`` has ``electron_lifetime_file`` which is not in ``proto_nd_flow``.


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

  Copied from ``yamls/proto_nd_flow/reco/charge/ExternalTriggerFinder.yaml``. No changes made.

* ``yamls/module1_flow/reco/charge/RawHitBuilder.yaml``

  Copied and modified from ``yamls/proto_nd_flow/reco/charge/RawHitBuilder.yaml``

  Included ``configuration_file`` and ``pedestal_file``, and set them to what I found in previously flowed file's metadata.

* ``yamls/module1_flow/reco/charge/EventBuilder.yaml``

   Copied from ``yamls/proto_nd_flow/reco/charge/EventBuilder.yaml``. No changes.


3. ``yamls/module1_flow/workflows/combined/combined_reconstruction.yaml``

   Copied and modified from ``yamls/proto_nd_flow/workflows/combined/combined_reconstruction.yaml``. Only difference is that the ``.yaml`` files point to ``module1_flow`` specific files. ``proto_nd_flow`` only had a ``t0_reco`` step, whlie ``module0_flow`` has ``drift_reco``, ``electron_lifetime_corr``, ``tracklet_reco``, ``tracklet_merge``.

* ``yamls/module1_flow/reco/combined/T0Reconstruction.yaml``

   Copied from ``yamls/proto_nd_flow/reco/combined/T0Reconstruction.yaml``. No changes made.

4. ``yamls/module1_flow/workflows/charge/prompt_calibration.yaml``

   Copied and modified from yamls/proto_nd_flow/workflows/charge/prompt_calibration.yaml. Only difference is that the ``.yaml`` files point to ``module1_flow`` specific files.

* ``yamls/module1_flow/reco/charge/CalibHitBuilder.yaml``

   Copied and modified from ``yamls/proto_nd_flow/reco/charge/CalibHitBuilder.yaml``.

   Added option for ``pedestal_file`` and ``configuration_file``, using inputs found in previoulsy flowed file metadata.

5. ``yamls/module1_flow/workflows/charge/final_calibration.yaml``

   Copied and modified from ``yamls/proto_nd_flow/workflows/charge/final_calibration.yaml``. Only difference is that ``.yaml`` files now point to ``module1_flow`` specific files.

* ``yamls/module1_flow/reco/charge/CalibHitMerger.yaml``

   Copied and modified from ``yamls/proto_nd_flow/workflows/charge/final_calibration.yaml``.

   Removed ``mc_hit_frac_dset_name``.
