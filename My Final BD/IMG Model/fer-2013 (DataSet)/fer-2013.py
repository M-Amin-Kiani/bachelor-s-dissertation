import pickle
from pathlib import Path
from typing import List

import datasets

logger = datasets.logging.get_logger(__name__)


_HOMEPAGE = "https://www.kaggle.com/datasets/msambare/fer2013"
_URL = "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/"
_URLS = {
    "train": _URL + "train.pt",
    "test": _URL + "test.pt",
}
_DESCRIPTION = "A large set of images of faces with seven emotional classes"
_CITATION = """\
@TECHREPORT{FER2013 dataset,
    author = {Prince Awuah Baffour},
    title = {Facial Emotion Detection},
    institution = {},
    year = {2022}
}
"""


class fer2013(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "img_bytes": datasets.Value("binary"),
                    "labels": datasets.features.ClassLabel(names=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]),
                }
            ),
            supervised_keys=("img_bytes", "labels"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"filepath": downloaded_files["train"]
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with Path(filepath).open("rb") as f:
            examples = pickle.load(f)

        for i, ex in enumerate(examples):
            yield str(i), ex
