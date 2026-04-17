# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-162`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b02e3d7c43b0542a9708c97a4decb5ab50a7fecdb19a413e8ba04a6c6f24587b`
- fixture hash: `sha256-fc0fa875ed0ba10ef61e5e8b6c1b783878d38dd1c5525b62b1d2717e4e66617b`
- score hash: `sha256-fa680a6b892536c76c4df06690205fc3af313f3260cbc66fb7a50db07fecd01f`
- bundle hash: `sha256-433a956f3d1b4f768aec9baa33d13b453165217532964e32616d7b99fbfd059d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f8f3d7baac7ea624c59c2785d2ad8b5f8904cda6bfe17f914b150feacd473265 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6a4fb3bfa8a04535b0185fefb9ab2abe8eb5373298ba391a39b9e783fac6c4f2 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-241afeab17343ca8ff976c8371860af95f601828c65f18c824cc157c9a20051c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-498e6da027655404f4aa03f835ed12b6ec793680643638f0b939c97e058e2c9d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e265b5ea | sha256-9abf9d7f4470fe5c9f044a75e36b8fe455d46050a61dbd90ca94d4ad4fc9da4c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e265b5ea | sha256-9b6578657c40163ca9e4451e631f6d927c035a1ac755a057a70a3e776177a6c5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3acefb97 | sha256-ce749b10408dd7cc127868e0b43ccfd8ca9e20cf11b75b5f6122f1110d4978cd |
