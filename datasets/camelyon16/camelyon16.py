"""camelyon16 dataset."""

import os, sys
from module.tfds.utils import Utils

from tensorflow import uint16
import tensorflow_datasets as tfds
import numpy as np

# TODO(camelyon16): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(camelyon16): BibTeX citation
_CITATION = """
"""


class Camelyon16(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for camelyon16 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(camelyon16): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'file_name': tfds.features.Text(),
            'image': tfds.features.Image(shape=(None, None, 3)),
            'segmentation_mask': tfds.features.Image(shape=(None, None, 1)),
            'color_region_mask': tfds.features.Image(shape=(None, None, 1)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'segmentation_mask'),  # Set to `None` to disable
        homepage='https://camelyon16.grand-challenge.org/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(camelyon16): Downloads the data and defines the splits
    url = Utils.get_gdrive_download_url('shared-url-here')
    extracted_dir = dl_manager.download_and_extract(url)

    # TODO(camelyon16): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(extracted_dir),
        'test': self._generate_examples(extracted_dir, 'test-'),
    }

  def _generate_examples(self, extracted_dir, prefix=''):
    data = Utils.extract_files_base_on_folder_from_extracted_dir(extracted_dir)
    data_pairs = Utils.pair_files_from_3_list(data[f'{prefix}image'], data[f'{prefix}target'], data['CR'])
    
    """Yields examples."""
    # TODO(camelyon16): Yields (key, example) tuples from the dataset
    for data in data_pairs:
      yield data['file_name'], data