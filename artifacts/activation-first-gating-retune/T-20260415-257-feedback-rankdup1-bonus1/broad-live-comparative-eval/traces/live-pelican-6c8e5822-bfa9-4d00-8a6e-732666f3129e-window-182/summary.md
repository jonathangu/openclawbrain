# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-182`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6cd785628fb3c34642dd7b4a701799a6e96acb06e347a7bf1d01cd4950a8de4c`
- fixture hash: `sha256-8169ace4aebbc5a4a546b5c0d2bdc7c5a395f1f1630a066be79c7f63594673d2`
- score hash: `sha256-a26c4c9303ce1e82b74e92c069129f9fec3a8902ef4a97ee4e136c6491a2b709`
- bundle hash: `sha256-8932b2e2c60aa66fe382defa088e028d22c4576f012746b47c0450ecb96dad60`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 1 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf6fbe7613b07ee3e659c5ab0ce2fe9e83640dc0dbe17b255f0c268784354a36 |
| vector_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-7670b2a10b499d3de55ac86612ba2df84b87582148e4db54c8c5b00f0552bf40 |
| graph_prior_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-0774e021ed38796a8d7a8cf0b96e2dd8384ac4bcc24be5d9ec493afbcf94e9b5 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-6143fca7ba79cabb711aa90b15034c2c11ef3b2bf1493016d9bf2a278838afb2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-e30c5d87 | sha256-1df203caceefa301a9929c3b53e60161dcfaee5e325b56f6962a2c36c413c661 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-e30c5d87 | sha256-7b32d4dbfba619b5adeebc4a13e3b8ba52c3072ee0175a3f6a5fb849498dd5fa |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-e30c5d87 | sha256-1df203caceefa301a9929c3b53e60161dcfaee5e325b56f6962a2c36c413c661 |
