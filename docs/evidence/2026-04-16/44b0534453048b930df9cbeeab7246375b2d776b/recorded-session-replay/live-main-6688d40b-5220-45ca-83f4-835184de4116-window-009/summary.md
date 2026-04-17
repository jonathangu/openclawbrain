# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-009`
- winner mode: `vector_only`
- trace hash: `sha256-1eeac1e61d3b2146227988d62e1fe6c84e0b77c1468137fa8c6d382736c2c4ff`
- fixture hash: `sha256-1569517b028e54a6250341eadd5d277f396164c98c963a62234840e80af05420`
- score hash: `sha256-5381817d71b1a8e7cc4bdd828b0e953fb9a9934dc3667d4187439dd07ab9e434`
- bundle hash: `sha256-d2f874f3e9c078eba3bd6ea4497395b7e2ab41ed4718deea22bfc1da82b586a3`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 60 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/12
- phrase hit rate: 0.083333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5439b064337936d23e3ce0669d08085a2f0dcec2b235478161f6d9e74cb033a |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-e483385ae95dfd4efda3edc945e4e10481fa29c02764ffbd7fb00eb4a1a3ee6a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b47b839459867d40208d7fbbaef5a7c4370fad4a89c3d2f9478216510cbc708 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b0f4776f700abf674fbd0914d94db4a7ee10af4b828fcb0428526894e9c3f4f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-8f69f756 | sha256-f03d8694fc6e64026b213a3ecd135be75dcf399aa14be4626338649df40d1e92 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f69f756 | sha256-4647cc2ef146ba9efe5406817bcb1bde527d4a5ad5a2e036918df7edf7cef138 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-02053043 | sha256-10ff4148183e0d7e6933b17d5c4bcba75f05e8d34910cd09a03f3b4821de3bf4 |
