# DATA: Candidates, Queries, and Topics

This directory contains essential data files for candidate generation, final queries, and dataset topics. These files are some of the outputs and resources used in our experiments. Below is an overview of its structure and purpose.

## Structure

- `candidates/`: Contains the outputs of candidate generation and ranking assessment processes.
- `queries/`: Stores final queries, such as the outputs from testing AdaRewriter.
- `topics/`: Includes test topics, training topics (datasets), and qrel files.

---

# Collection: iKAT and QReCC

We utilize two main collections: iKAT (ClueWeb22-B) and QReCC (Common Crawl and the Wayback Machine).

## iKAT Collection

The iKAT text collection is a subset of ClueWeb22 documents, prepared by the organizers in collaboration with CMU. These documents have been split into ~116M passages. The goal is to retrieve passages from target open-domain text collections.

### License for ClueWeb22-B

Getting the license to use the collection can be time-consuming and would be handled by CMU, not the iKAT organizers. Please follow these steps to get your data license:

1. Sign the license form available on the ClueWeb22 project [webpage](https://lemurproject.org/clueweb22/ClueWeb22%20Organization%20License%20(06.29.22.2).pdf) and send the form to CMU for approval (clueweb@andrew.cmu.edu).
2. Once you have the license, email Andrew Ramsay (andrew.ramsay@glasgow.ac.uk) to request access to a download link for the preprocessed iKAT passage collection and other resources such as Lucene and SPLADE indexes.

> **Note**: CMU requires a signature from the organization (e.g., university or company), not an individual. Please give enough time to the CMU licensing office to accept your request.

## QReCC Collection

To use the QReCC collection:

1. Download the collection:

    ```bash
    wget https://zenodo.org/records/5760304/files/passages.zip
    ```

2. Index the passages using Pyserini, a Python wrapper around Anserini. Java (JDK) is required as a prerequisite. After installing Pyserini, use the following command to build the index:

    ```bash
    time python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
     -threads 76 -input collection-paragraph \
     -index index-paragraph -storePositions -storeDocvectors -storeRaw
    ```

    > **Note**: For us, this took less than 2 hours.

## Citation

### iKAT-Baseline

```bibtex
@inproceedings{coordinators-trec2024-papers-proc-4,
    author = {Mohammad Aliannejadi (University of Amsterdam), Zahra Abbasiantaeb (University of Amsterdam), Simon Lupart (University of Amsterdam), Shubham Chatterjee (University of Edinburgh), Jeffrey Dalton (University of Edinburgh), Leif Azzopardi (University of Strathclyde)},
    title = {TREC iKAT 2024: The Interactive Knowledge Assistance Track Overview},
    booktitle = {The Thirty-Third Text REtrieval Conference Proceedings (TREC 2024), Gaithersburg, MD, USA, November 15-18, 2024},
    series = {NIST Special Publication},
    volume = {1329},
    publisher = {National Institute of Standards and Technology (NIST)},
    year = {2024},
    trec_org = {coordinators},
    trec_runs = {},
    trec_tracks = {ikat},
    url = {https://trec.nist.gov/pubs/trec33/papers/Overview_ikat.pdf}
}
```

The original iKAT-Baseline source code is available at: [https://github.com/SimonLupart/ikat-baseline](https://github.com/SimonLupart/ikat-baseline)

### QReCC

```bibtex
@article{qrecc,
  title={Open-Domain Question Answering Goes Conversational via Question Rewriting},
  author={Anantha, Raviteja and Vakulenko, Svitlana and Tu, Zhucheng and Longpre, Shayne and Pulman, Stephen and Chappidi, Srinivas},
  journal={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2021}
}
```

The original QReCC source code is available at: [https://github.com/apple/ml-qrecc](https://github.com/apple/ml-qrecc)

### SCAI-QReCC-21

```bibtex
@misc{vakulenko2022scaiqreccsharedtaskconversational,
      title={SCAI-QReCC Shared Task on Conversational Question Answering}, 
      author={Svitlana Vakulenko and Johannes Kiesel and Maik Fröbe},
      year={2022},
      eprint={2201.11094},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2201.11094}, 
}
```

The original SCAI-QReCC-21 source code is available at: [https://github.com/scai-conf/SCAI-QReCC-21](https://github.com/scai-conf/SCAI-QReCC-21)
