# Fresh CUDA release check

These lossless files archive the separate fresh CUDA run recorded in the section verification report on 2026-09-05. Every original JSON SHA256 was checked against that report before compression. The manifest records original and compressed hashes; the unchanged report is included as gzip. Its original temporary directory remains part of the authentic run record.

The fresh model refit checks all 65 training-fitted target/null score directions, and baseline plus 34 unit-strength intervention conditions on all 104 final inputs: 3,640 generated examples. All saved individual outcomes match the completed learner exactly. This is independent regeneration, not a replay of stored output. Actual run time is 173.606 seconds with peak memory 1.9182 GiB. Sensitive rows contain completion hashes and outcomes; concrete benign completions and tokens remain inspectable. Marker absence is not correctness or safety.

The notebook itself generated all 15 development candidates and all 85 final conditions in its own full run. This independent subset recheck supports that learner evidence; it does not replace it or claim a full replication of the paper's model and evaluation suite.
