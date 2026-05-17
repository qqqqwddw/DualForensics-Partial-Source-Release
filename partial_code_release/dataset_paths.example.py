# Example dataset-path configuration for evaluation.
# Copy this file to dataset_paths.py and replace the paths with your local datasets.

DATASET_PATHS = [
    dict(
        real_path='/path/to/real/images',
        fake_path='/path/to/fake/images',
        data_mode='wang2020',
        key='example_generator',
    ),
]
