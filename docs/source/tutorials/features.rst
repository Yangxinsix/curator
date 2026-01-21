Feature workflows
=================

This page shows how to compute features, cache them, and use distance metrics
with the feature stack in ``curator.layer._feature``.

Quick start: dataset features
-----------------------------

Use ``FeatureStatistics`` to compute features for a dataset and a list of models.
The output is a dict keyed by kernel name, with tensors shaped ``m x n x p``.

.. code-block:: python
    :linenos:

    from curator.data import AseDataset
    from curator.layer._feature import FeatureStatistics

    dataset = AseDataset("train.traj")
    stats = FeatureStatistics(
        models=[model],
        dataset=dataset,
        kernels=[("full-g", 512)],
        batch_size=8,
    )
    features = stats.get_features(normalize=True)
    full_features = features["full-gradient"]

Notes:
- Kernel aliases like ``full-g`` are normalized to ``full-gradient`` in outputs.
- Use ``local-full-g`` or ``local-gnn`` for local kernels.

Checkpointing with HDF5
-----------------------

For large datasets or resume support, use ``H5Feature`` as a cache:

.. code-block:: python
    :linenos:

    from curator.layer._feature import FeatureStatistics, H5Feature

    store = H5Feature("features.h5", num_models=len(models))
    stats = FeatureStatistics(
        models=models,
        dataset=dataset,
        kernels=[("full-g", 512)],
        store=store,
        checkpoint_interval=10,
    )
    features = stats.get_features(normalize=False)

Distance metrics (Mahalanobis)
------------------------------

``DistanceMetrics`` operates on feature tensors and computes statistics for
Mahalanobis distance (mean/std/precision).

.. code-block:: python
    :linenos:

    from curator.layer._feature import DistanceMetrics

    metrics = DistanceMetrics(regularization=1e-6)
    metrics.fit(features["full-gradient"])
    dist = metrics.score(features["full-gradient"])

Attach distance output to a model
---------------------------------

If you want a model to emit Mahalanobis distance during inference, attach a
``FeatureCalculator`` to ``model.output_modules``.

.. code-block:: python
    :linenos:

    from curator.layer._feature import FeatureCalculator

    calc = FeatureCalculator(
        kernels=[("full-g", 512)],
        compute_maha_dist=True,
        dataset="train.traj",
        distance_kernel="full-g",
        output_features=False,
    )
    model.output_modules.append(calc)

The calculator computes distance statistics once (from ``dataset``) and then
outputs ``properties.maha_dist`` in each forward pass. Set ``output_features``
``True`` if you also want to keep ``properties.feature`` in the model outputs.

Active learning selection
-------------------------

Use ``GeneralActiveLearning`` to compute features, build a kernel matrix, and
select structures.

.. code-block:: python
    :linenos:

    from curator.select import GeneralActiveLearning

    selector = GeneralActiveLearning(
        models=models,
        kernels=[("full-g", 512)],
    )
    result = selector.select(
        pool_set="pool.traj",
        train_set="train.traj",
        n_select=100,
    )
